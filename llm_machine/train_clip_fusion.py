#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import json
import time
import random
from typing import Dict, Any, Tuple, List, Optional
import argparse
import glob
from collections import defaultdict
import re
from tqdm import tqdm
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode

# ====== your modules ======
from llm_machine import LLMTextEncoder
from llm_machine import VisionTextSigLIP
from llm_machine.data_linked import ImageDatasetMultiLabel, ImageCollatorMulti, TextCollatorSingle
from networks import Token
from llm_machine import train_step_linked  # <-- 당신 코드 기준: train_step_linked(vt, batch_img, batch_txt, optimizer, scaler)

# ===================== #
#   기본 설정 / 경로     #
# ===================== #
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

image_size = (1024, 512)
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD  = [0.26862954, 0.26130258, 0.27577711]

ITEMS_IMG_PATH = "/home/policelab_l40s/llm_prompt/llm_prompt/json_file/shoerinics_group_gt.jsonl"
ITEMS_TXT_PATH = "/home/policelab_l40s/llm_prompt/llm_prompt/json_file/최종_txt_multimodal_train.jsonl"

CKPT_DIR = "/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip"
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 600
BATCH_SIZE_IMG = 30
NUM_WORKERS_IMG = 4

LR = 1e-5
WEIGHT_DECAY = 0.01

LABEL_HIT_AT_K = 5
LOG_EVERY_STEPS = 10

DEBUG_BACKPROP = True
DEBUG_SAMPLE_PARAMS = 3
DEBUG_EVERY_STEPS = 1


# ===================== #
#        DDP Utils      #
# ===================== #
def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def setup_ddp_from_torchrun():
    """
    torchrun 사용 시:
      - RANK, WORLD_SIZE, LOCAL_RANK 환경변수가 주입된다.
      - init_process_group(backend=nccl)로 초기화한다.
      - local_rank에 맞춰 cuda device를 고정한다.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not ddp_is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return device, local_rank, True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, False

def cleanup_ddp():
    if ddp_is_initialized():
        dist.destroy_process_group()

def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()


# ===================== #
#     유틸 함수 묶음     #
# ===================== #
def _now():
    return time.strftime("%H:%M:%S")

def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

def log0(msg: str):
    if is_main_process():
        print(f"[{_now()}] {msg}", flush=True)

@contextmanager
def timeit0(tag: str):
    t0 = time.perf_counter()
    log0(f"▶ {tag} ...")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log0(f"✔ {tag} done in {format_eta(dt)}")

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def _safe_torch_load(ckpt_path: str, map_location="cpu"):
    try:
        import numpy as np
        from torch.serialization import safe_globals
        with safe_globals([np.core.multiarray.scalar]):
            try:
                return torch.load(ckpt_path, map_location=map_location)
            except TypeError:
                return torch.load(ckpt_path, map_location=map_location)
    except Exception:
        try:
            return torch.load(ckpt_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(ckpt_path, map_location=map_location)

def _unwrap_state_dict(maybe_wrapped: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("state_dict", "model", "net", "module"):
        if key in maybe_wrapped and isinstance(maybe_wrapped[key], dict):
            return maybe_wrapped[key]
    return maybe_wrapped

def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

def _select_by_prefix_and_shape(
    checkpoint: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
    allowed_prefixes: Tuple[str, ...] = ("backbone", "tr"),
) -> Dict[str, torch.Tensor]:
    filtered = {}
    for k, v in checkpoint.items():
        if not any(k.startswith(pfx) for pfx in allowed_prefixes):
            continue
        if (k in model_state) and (model_state[k].shape == v.shape):
            filtered[k] = v
    return filtered

def save_weights(vt, token_model, epoch, global_step, tag="last"):
    vt_core = vt.module if hasattr(vt, "module") else vt
    vt_sd = vt_core.state_dict()

    token_core = token_model.module if hasattr(token_model, "module") else token_model
    token_sd = token_core.state_dict()

    vt_path = os.path.join(CKPT_DIR, f"vt_multimodal_clip_no_freeze{tag}.pt")
    token_path = os.path.join(CKPT_DIR, f"token_multimodal_clip_no_freeze{tag}.pt")

    torch.save(vt_sd, vt_path)
    torch.save(token_sd, token_path)

    log0(f"[Save] vt    -> {vt_path}")
    log0(f"[Save] token -> {token_path}")

def load_vt_from_path(vt: torch.nn.Module, path: str, map_location="cpu", strict: bool = False):
    assert os.path.isfile(path), f"vt ckpt not found: {path}"
    raw = _safe_torch_load(path, map_location=map_location)
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
    else:
        sd = _unwrap_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")
    vt.load_state_dict(sd, strict=strict)
    log0(f"[LoadVT] loaded vt weights from {path}")

def load_token_from_path(token_model: torch.nn.Module, path: str, map_location="cpu", strict: bool = False):
    assert os.path.isfile(path), f"token ckpt not found: {path}"
    raw = _safe_torch_load(path, map_location=map_location)
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
    else:
        sd = _unwrap_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")

    model_wo_ddp = token_model.module if hasattr(token_model, "module") else token_model
    model_dict = model_wo_ddp.state_dict()
    filtered = _select_by_prefix_and_shape(sd, model_dict, allowed_prefixes=("backbone", "tr"))
    if len(filtered) == 0:
        filtered = {
            k: v for k, v in sd.items()
            if (k in model_dict) and (model_dict[k].shape == v.shape)
            and not any(x in k for x in ["classifier", "fc", "head", "heads", "arcface"])
        }

    incompatible = model_wo_ddp.load_state_dict(filtered, strict=False if not strict else True)
    miss = list(getattr(incompatible, "missing_keys", []))
    unexp = list(getattr(incompatible, "unexpected_keys", []))

    log0(f"[LoadToken] loaded {len(filtered)} keys from {path}")
    if miss:
        log0(f"[LoadToken] Missing keys: {len(miss)} (first 10): {miss[:10]}")
    if unexp:
        log0(f"[LoadToken] Unexpected keys: {len(unexp)} (first 10): {unexp[:10]}")

def _gather_paths(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    parts: List[str] = []
    for token in arg.replace(",", " ").split():
        if any(ch in token for ch in "*?[]"):
            parts.extend(glob.glob(token))
        else:
            parts.append(token)
    out = []
    seen = set()
    for p in parts:
        p = os.path.abspath(p)
        if (p not in seen) and os.path.isfile(p):
            out.append(p); seen.add(p)
    return out


# ===================== #
#   라벨별 텍스트 준비    #
# ===================== #
def build_label2texts(items_txt: List[Dict[str, Any]]) -> Dict[int, List[str]]:
    table: Dict[int, List[str]] = defaultdict(list)
    for it in items_txt:
        lab = int(it["label"])
        table[lab].append(it["text"])
    return table

def _unique_labels_in_batch(batch_img) -> List[int]:
    if hasattr(batch_img, "label_sets"):
        uniq = set()
        for labs in batch_img.label_sets:
            for lab in labs:
                uniq.add(int(lab))
        return sorted(uniq)

    if isinstance(batch_img, dict) and "labels" in batch_img:
        labels = batch_img["labels"]
        if torch.is_tensor(labels) and labels.dim() == 2:
            uniq = set()
            for i in range(labels.size(0)):
                idxs = labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
                uniq.update(map(int, idxs))
            return sorted(uniq)
        else:
            uniq = set()
            for labs in labels:
                uniq.update(map(int, labs))
            return sorted(uniq)

    raise TypeError("batch_img에서 라벨 정보를 찾을 수 없다. (label_sets 또는 labels 필요)")

def _build_text_batch_one_per_label(
    batch_img,
    label2texts: Dict[int, List[str]],
    tokenizer,
    max_length: int = 128,
):
    uniq_labels = _unique_labels_in_batch(batch_img)
    items_for_collate: List[Dict[str, Any]] = []
    label_ids_in_text_order: List[int] = []
    for lab in uniq_labels:
        cand = label2texts.get(lab, [])
        if not cand:
            continue
        txt = random.choice(cand)
        items_for_collate.append({"text": txt, "label": int(lab)})
        label_ids_in_text_order.append(int(lab))

    if len(items_for_collate) == 0:
        return None

    collate = TextCollatorSingle(tokenizer, max_length=max_length)
    batch_txt = collate(items_for_collate)
    setattr(batch_txt, "label_ids", label_ids_in_text_order)
    return batch_txt


# ===================== #
#    역전파 계측 유틸     #
# ===================== #
class GradProbe:
    def __init__(self, model: torch.nn.Module, name_prefix: str = "token"):
        self.model = model
        self.name_prefix = name_prefix
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.reset()

    def attach(self):
        self.detach()
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            handle = p.register_hook(self._make_hook(name))
            self._handles.append(handle)

    def detach(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def reset(self):
        self.had_any_grad = False
        self.grad_param_count = 0
        self.total_grad_norm_sq = 0.0

    def _make_hook(self, name: str):
        def _hook(grad: torch.Tensor):
            if grad is None:
                return
            self.had_any_grad = True
            self.grad_param_count += 1
            with torch.no_grad():
                self.total_grad_norm_sq += float(grad.detach().pow(2).sum().cpu().item())
        return _hook

    def summary(self) -> Dict[str, Any]:
        total_grad_norm = (self.total_grad_norm_sq ** 0.5)
        return {
            "had_any_grad": bool(self.had_any_grad),
            "grad_param_count": int(self.grad_param_count),
            "grad_total_norm": float(total_grad_norm),
        }

class ParamSnapshot:
    def __init__(self, model: torch.nn.Module, max_params: int = 3):
        self.model = model
        self.max_params = max_params
        self.before: List[Tuple[str, torch.Tensor]] = []
        self.after:  List[Tuple[str, torch.Tensor]] = []

    def take_before(self):
        self.before = []
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.data.numel() >= 1024:
                self.before.append((name, p.data.detach().cpu().flatten()[:32].clone()))
                if len(self.before) >= self.max_params:
                    break

    def take_after(self):
        self.after = []
        named = dict(self.model.named_parameters())
        for name, _snap in self.before:
            p = named[name]
            self.after.append((name, p.data.detach().cpu().flatten()[:32].clone()))

    def changed_flags(self) -> List[bool]:
        flags = []
        for (n1, b), (n2, a) in zip(self.before, self.after):
            flags.append(bool(not torch.equal(b, a)))
        return flags


# ===================== #
#  멀티라벨 히트 비율     #
# ===================== #
def _extract_image_label_sets(batch_img) -> List[set]:
    if hasattr(batch_img, "label_sets"):
        return [set(map(int, labs)) for labs in batch_img.label_sets]

    if isinstance(batch_img, dict) and "labels" in batch_img:
        labels = batch_img["labels"]
        if torch.is_tensor(labels):
            if labels.dim() == 2:
                out = []
                for i in range(labels.size(0)):
                    idxs = labels[i].nonzero(as_tuple=False).squeeze(1).tolist()
                    out.append(set(map(int, idxs)))
                return out
            else:
                raise TypeError("labels 텐서는 [B, C] multi-hot 형태여야 한다.")
        else:
            return [set(map(int, labs)) for labs in labels]

    raise TypeError("batch_img에서 라벨 정보를 찾을 수 없다. (label_sets 또는 labels 필요)")

@torch.no_grad()
def _compute_label_hit_ratio_at_k(
    vt,  # DDP일 수도 있음
    images_t: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_label_sets: List[set],
    text_label_ids: List[int],
    k: int = 5,
) -> Tuple[float, int, int]:
    vt_core = vt.module if hasattr(vt, "module") else vt
    device = next(vt_core.parameters()).device
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        vt_core.eval()
        out = vt_core(images_t, input_ids, attention_mask, targets=None)
        vt_core.train()

    candidate_keys = ["logits", "logits_per_image", "sims", "similarity", "scores"]
    scores = None
    if isinstance(out, dict):
        for kk in candidate_keys:
            if kk in out:
                scores = out[kk]
                break
    elif torch.is_tensor(out):
        scores = out

    if scores is None:
        raise RuntimeError("VisionTextSigLIP forward 결과에서 점수 행렬 키를 찾지 못했다.")

    B, M = scores.shape
    kk = min(k, M)
    _, topk_idx = torch.topk(scores, k=kk, dim=1)

    text_label_ids_t = torch.tensor(text_label_ids, device=scores.device)
    topk_label_ids = text_label_ids_t[topk_idx].cpu().tolist()

    total_hits = 0
    total_gt = 0
    for i in range(B):
        gt = image_label_sets[i]
        if len(gt) == 0:
            continue
        preds_k = set(topk_label_ids[i])
        hits = len(preds_k.intersection(gt))
        total_hits += hits
        total_gt += len(gt)

    ratio = (total_hits / total_gt * 100.0) if total_gt > 0 else float("nan")
    return ratio, total_hits, total_gt


# ===================== #
#      모델 빌드          #
# ===================== #
def build_model(
    device: torch.device,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False,
    use_ddp: bool = False,
    find_unused_parameters: bool = True,  # <-- 이전 DDP unused grad 에러 방지용 (일단 True 권장)
):
    token_model = Token(outputdim=1024, classifier_num=81313, mode='train').to(device)

    # ※ 당신 원본대로: backbone 포함 전부 freeze 상태(학습 파라미터 거의 없을 수 있음)
    for name, p in token_model.named_parameters():
        if name.startswith("backbone"):
            p.requires_grad = True
        else:
            p.requires_grad = True

    token_model.train()

    if token_ckpt_path:
        try:
            load_token_from_path(token_model, token_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            log0(f"[Token] Failed to load TOKEN from '{token_ckpt_path}': {e}")

    text_encoder = LLMTextEncoder(
        model_name=MODEL_NAME,
        device=str(device),
        dtype=torch.bfloat16,
        train_llm=True,
        use_lora=True,
        lora_r=8, lora_alpha=16, lora_dropout=0.1,
        pooling="mean",
    )

    vt = VisionTextSigLIP(
        token_model=token_model,
        text_encoder=text_encoder,
        vision_dim=1024,
        proj_out_dim=1024,
        temperature_init=0.06,
    ).to(device).train()

    if vt_ckpt_path:
        try:
            load_vt_from_path(vt, vt_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            log0(f"[VT] Failed to load VT from '{vt_ckpt_path}': {e}")

    if use_ddp and ddp_is_initialized():
        vt = DDP(
            vt,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )

    return vt, token_model


# ===================== #
#       학습 루프         #
# ===================== #
def main_train(
    image_size=image_size,
    norm_mean=tuple(NORM_MEAN),
    norm_std=tuple(NORM_STD),
    white_bg_fill=True,
    allow_flip=False,
    save_interval=10,
    resume_tag: Optional[str] = None,
    resume_path: Optional[str] = None,
    resume_all: bool = False,  # (현재 코드에서는 optimizer/scaler resume는 구현하지 않음)
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False,
):
    device, local_rank, use_ddp = setup_ddp_from_torchrun()
    torch.backends.cudnn.benchmark = (device.type == "cuda")

    log0(f"[DDP] use_ddp={use_ddp} | world_size={get_world_size()} | rank={get_rank()} | local_rank={local_rank} | device={device}")

    # 1) 모델 구성
    vt, token_model = build_model(
        device=device,
        vt_ckpt_path=vt_ckpt_path,
        token_ckpt_path=token_ckpt_path,
        strict_load=strict_load,
        use_ddp=use_ddp,
        find_unused_parameters=True,
    )

    vt_core = vt.module if hasattr(vt, "module") else vt

    # 2) 데이터 로드
    assert os.path.isfile(ITEMS_IMG_PATH), f"not found: {ITEMS_IMG_PATH}"
    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"

    with timeit0("Load JSONL"):
        items_img = load_jsonl(ITEMS_IMG_PATH)
        items_txt = load_jsonl(ITEMS_TXT_PATH)
        label2texts = build_label2texts(items_txt)

    tfm = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomApply([
            T.RandomAffine(
                degrees=2,
                translate=(0.01, 0.01),
                scale=(0.98, 1.02),
                shear=1,
                interpolation=InterpolationMode.BICUBIC,
                fill=255 if white_bg_fill else 0
            )
        ], p=0.30),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.02, p=1.0)], p=0.10),
        T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05)], p=0.30),
        T.RandomHorizontalFlip(p=0.15 if allow_flip else 0.0),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.15),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    img_ds = ImageDatasetMultiLabel(items_img, image_transform=tfm)

    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            img_ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False
        )

    img_loader = DataLoader(
        img_ds,
        batch_size=BATCH_SIZE_IMG,                  # per-GPU batch
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS_IMG,
        collate_fn=ImageCollatorMulti(),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS_IMG > 0),
        drop_last=False,
    )

    steps_per_epoch = len(img_loader)
    num_images = len(img_ds)

    log0(f"[Data] #images={num_images} | batch_size(perGPU)={BATCH_SIZE_IMG} | world_size={get_world_size()} | global_batch={BATCH_SIZE_IMG * get_world_size()} | steps/epoch={steps_per_epoch}")

    # 3) 옵티마이저 & AMP
    trainable = [p for p in vt_core.parameters() if p.requires_grad]
    if len(trainable) == 0:
        log0("[WARN] trainable 파라미터가 0개입니다. (현재 코드에서는 token_model/기타가 전부 freeze 상태입니다.)")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    tokenizer = vt_core.text_encoder.tokenizer

    # 계측 도구
    grad_probe = GradProbe(token_model, name_prefix="token")
    if DEBUG_BACKPROP:
        grad_probe.attach()
    snap = ParamSnapshot(token_model, max_params=DEBUG_SAMPLE_PARAMS)

    total_steps = EPOCHS * steps_per_epoch
    ema_iter_time = None
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        if use_ddp and sampler is not None:
            sampler.set_epoch(epoch)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        epoch_start = time.perf_counter()
        cum_hits = 0
        cum_gt = 0

        for step, batch_img in enumerate(img_loader, start=1):
            t0 = time.perf_counter()

            batch_txt = _build_text_batch_one_per_label(
                batch_img=batch_img,
                label2texts=label2texts,
                tokenizer=tokenizer,
                max_length=64
            )
            if batch_txt is None:
                continue

            if DEBUG_BACKPROP and (global_step % DEBUG_EVERY_STEPS == 0):
                grad_probe.reset()
                snap.take_before()

            # =======================
            #    핵심: train step
            # =======================
            logs = train_step_linked(vt, batch_img, batch_txt, optimizer, scaler)
            cur_loss = float(logs.get("arcface_loss", logs.get("loss", float("nan"))))
            cur_temp = float(logs.get("temp", float("nan")))

            if DEBUG_BACKPROP and (global_step % DEBUG_EVERY_STEPS == 0):
                snap.take_after()
                _ = snap.changed_flags()
                _ = grad_probe.summary()

            # 라벨 히트 비율 측정(옵션)
            text_label_ids = getattr(batch_txt, "label_ids", None)
            batch_ratio = float("nan")
            if text_label_ids is not None:
                img_label_sets = _extract_image_label_sets(batch_img)

                if hasattr(batch_img, "images"):
                    images_t = batch_img.images
                elif isinstance(batch_img, dict) and "images" in batch_img:
                    images_t = batch_img["images"]
                else:
                    raise TypeError("batch_img에서 images 텐서를 찾을 수 없다.")

                images_t = images_t.to(device, non_blocking=True)
                input_ids = batch_txt.input_ids.to(device, non_blocking=True)
                attention_mask = batch_txt.attention_mask.to(device, non_blocking=True)

                ratio, hits, gt = _compute_label_hit_ratio_at_k(
                    vt=vt,
                    images_t=images_t,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_label_sets=img_label_sets,
                    text_label_ids=text_label_ids,
                    k=LABEL_HIT_AT_K,
                )
                batch_ratio = ratio
                cum_hits += hits
                cum_gt += gt

            global_step += 1

            iter_time = time.perf_counter() - t0
            ema_iter_time = iter_time if ema_iter_time is None else (0.9 * ema_iter_time + 0.1 * iter_time)

            steps_done = (epoch - 1) * steps_per_epoch + step
            steps_left = max(0, total_steps - steps_done)
            eta_sec = steps_left * (ema_iter_time if ema_iter_time is not None else iter_time)

            lr_cur = optimizer.param_groups[0]["lr"]
            max_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0

            if (step % LOG_EVERY_STEPS) == 0 and is_main_process():
                cum_ratio = (cum_hits / cum_gt * 100.0) if cum_gt > 0 else float("nan")
                print(
                    f">> Train Epoch: [{epoch}] "
                    f"[{step}/{steps_per_epoch}] "
                    f"eta: {format_eta(eta_sec)} "
                    f"loss: {cur_loss:.4f} "
                    f"Label-Hit@{LABEL_HIT_AT_K}: batch {batch_ratio:6.3f}% | epoch {cum_ratio:6.3f}% "
                    f"iter: {iter_time:.4f}s "
                    f"lr: {lr_cur:.2e} "
                    f"max_mem: {int(max_mem_mb)}MB",
                    flush=True
                )

        epoch_time = time.perf_counter() - epoch_start
        log0(f">> Epoch [{epoch}] done in {format_eta(epoch_time)}")

        # 저장: rank0만
        if (epoch % save_interval) == 0:
            ddp_barrier()
            if is_main_process():
                save_weights(vt, token_model, epoch, global_step, tag=f"epoch{epoch:03d}")
            ddp_barrier()

    ddp_barrier()
    if is_main_process():
        save_weights(vt, token_model, EPOCHS, global_step, tag="last")
        log0(">> Training done")


# ===================== #
#       추론 유틸         #
# ===================== #
@torch.no_grad()
def _forward_scores(vt_core, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    vt_core.eval()
    out = vt_core(images, input_ids, attention_mask, targets=None)
    candidate_keys = ["logits", "logits_per_image", "sims", "similarity", "scores"]
    scores = None
    if isinstance(out, dict):
        for k in candidate_keys:
            if k in out:
                scores = out[k]
                break
    elif torch.is_tensor(out):
        scores = out
    if scores is None:
        raise RuntimeError("VisionTextSigLIP forward 결과에서 점수 행렬 키를 찾지 못했다.")
    return scores

def _topk_unique_by_label(scores_1d, labels_of_texts: List[int], k: int, min_score: Optional[float] = None):
    if torch.is_tensor(scores_1d):
        scores_np = scores_1d.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores_1d, dtype=np.float32)

    order = np.argsort(-scores_np)
    chosen_idx, chosen_scores = [], []
    seen_labels = set()

    for j in order:
        s = float(scores_np[j])
        lab = int(labels_of_texts[j])
        if lab in seen_labels:
            continue
        if (min_score is not None) and (s < min_score):
            continue
        seen_labels.add(lab)
        chosen_idx.append(int(j))
        chosen_scores.append(s)
        if len(chosen_idx) >= k:
            break

    return chosen_idx, chosen_scores

@torch.no_grad()
def run_infer(
    topk: int = 5,
    text_batch: int = 256,
    image_paths: Optional[List[str]] = None,
    max_demo_images: int = 8,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False,
    fused_outdir: Optional[str] = None,
    infer_batch_size: int = 64,
):
    # infer는 DDP 사용하지 않는 것이 일반적이므로 단일 GPU/단일 프로세스로 수행
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vt, token_model = build_model(
        device=device,
        vt_ckpt_path=vt_ckpt_path,
        token_ckpt_path=token_ckpt_path,
        strict_load=strict_load,
        use_ddp=False,
    )
    vt_core = vt

    vt_core.eval()

    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_txt = load_jsonl(ITEMS_TXT_PATH)
    texts = [it["text"] for it in items_txt]
    labels_of_texts = [int(it["label"]) for it in items_txt]
    print(f"[Infer] #texts (candidates): {len(texts)}")

    if not image_paths:
        assert os.path.isfile(ITEMS_IMG_PATH), f"not found: {ITEMS_IMG_PATH}"
        items_img = load_jsonl(ITEMS_IMG_PATH)
        image_paths = [it["image_path"] for it in items_img][:max_demo_images]
    print(f"[Infer] #images: {len(image_paths)}")

    tfm = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    image_tensors, valid_paths = [], []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                image_tensors.append(tfm(im))
                valid_paths.append(p)
        except Exception as e:
            print(f"[Warn] 이미지 열기 실패: {p} ({e})")

    if len(image_tensors) == 0:
        print("[Infer] 사용할 이미지가 없습니다.")
        return

    num_imgs = len(image_tensors)
    print(f"[Infer] 유효 이미지 수: {num_imgs}")

    # 1) 텍스트 임베딩 전체 계산
    proj_dim = vt_core.proj_out_dim if hasattr(vt_core, "proj_out_dim") else 1024
    text_embs_all = np.zeros((len(texts), proj_dim), dtype=np.float32)

    collate = TextCollatorSingle(vt_core.text_encoder.tokenizer, max_length=128)

    t_start = 0
    while t_start < len(texts):
        t_end = min(t_start + text_batch, len(texts))
        cur_items = [{"text": t, "label": 0} for t in texts[t_start:t_end]]
        batch = collate(cur_items)

        input_ids = batch.input_ids.to(device, non_blocking=True)
        attention_mask = batch.attention_mask.to(device, non_blocking=True)

        use_amp = (device.type == "cuda")
        amp_dtype = torch.bfloat16

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            t_emb = vt_core.encode_texts(input_ids, attention_mask)

        text_embs_all[t_start:t_end, :] = t_emb.detach().cpu().to(torch.float32).numpy()
        t_start = t_end

    # 2) 이미지 배치 × 텍스트 배치 점수 계산
    fused_vecs = []
    meta_all = {}

    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16

    for img_start in tqdm(range(0, num_imgs, infer_batch_size), desc="Processing image batches"):
        img_end = min(img_start + infer_batch_size, num_imgs)
        batch_imgs = torch.stack(image_tensors[img_start:img_end], dim=0).to(device, non_blocking=True)
        B = batch_imgs.size(0)

        all_scores_chunk = torch.empty((B, len(texts)), dtype=torch.float32, device=device)

        t_start = 0
        while t_start < len(texts):
            t_end = min(t_start + text_batch, len(texts))
            cur_items = [{"text": t, "label": 0} for t in texts[t_start:t_end]]
            batch = collate(cur_items)

            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                scores = _forward_scores(vt_core, batch_imgs, input_ids, attention_mask)

            all_scores_chunk[:, t_start:t_end] = scores
            t_start = t_end

        all_scores_cpu = all_scores_chunk.detach().cpu()

        for bi, global_idx in enumerate(range(img_start, img_end)):
            img_path = valid_paths[global_idx]
            scores_i = all_scores_cpu[bi]

            chosen_idx, weights = _topk_unique_by_label(
                scores_1d=scores_i,
                labels_of_texts=labels_of_texts,
                k=topk,
                min_score=0.0,
            )

            if len(chosen_idx) == 0:
                print(f"[Fuse] skip (no positive scores): {img_path}")
                continue

            W = np.asarray(weights, dtype=np.float32)
            V = text_embs_all[chosen_idx, :]
            W = W / (W.sum() + 1e-12)
            fused_vec = (W[:, None] * V).sum(axis=0).astype(np.float32)
            fused_vecs.append(fused_vec)

            meta_all[os.path.basename(img_path)] = [
                {
                    "rank": r + 1,
                    "text_index": int(ti),
                    "label": int(labels_of_texts[ti]),
                    "score": float(weights[r]),
                    "text": texts[ti],
                }
                for r, ti in enumerate(chosen_idx)
            ]

    if len(fused_vecs) == 0:
        print("[Fuse] fused_vecs가 비어 있습니다 (모든 이미지 skip됨).")
        return

    fused_vecs = np.stack(fused_vecs, axis=0)

    if fused_outdir is None or len(str(fused_outdir).strip()) == 0:
        fused_outdir = os.path.join(os.path.dirname(valid_paths[0]), "fused_vecs")
    os.makedirs(fused_outdir, exist_ok=True)

    npy_path = os.path.join(fused_outdir, "all_images_fused_topk_unique.npy")
    np.save(npy_path, fused_vecs)

    meta_path = os.path.join(fused_outdir, "all_images_fused_topk_unique.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print(f"[Fuse] saved one npy for all images: {npy_path} (shape={fused_vecs.shape})")
    print(f"[Fuse] saved meta json: {meta_path}")


# ===================== #
#    엔트리 포인트       #
# ===================== #
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train",
                    choices=["train", "infer", "both"],
                    help="train: 학습, infer: 데모 추론, both: 학습 후 데모 추론")

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--text_batch", type=int, default=256)
    ap.add_argument("--infer_images", type=str, default="")
    ap.add_argument("--max_demo_images", type=int, default=1208)
    ap.add_argument("--infer_batch_size", type=int, default=64)

    ap.add_argument("--vt_ckpt_path", type=str,
                    default="/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/vt_multimodal_clip_no_freezeepoch010.pt")
    ap.add_argument("--token_ckpt_path", type=str,
                    default="/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/token_multimodal_clip_no_freezeepoch010.pt")

    ap.add_argument("--strict_load", type=int, default=0)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()

    try:
        if args.mode in ("train", "both"):
            main_train(
                vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
                token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
                strict_load=bool(args.strict_load),
            )

        if args.mode in ("infer", "both"):
            paths = _gather_paths(args.infer_images)
            run_infer(
                topk=args.topk,
                text_batch=args.text_batch,
                image_paths=paths if len(paths) > 0 else None,
                max_demo_images=args.max_demo_images,
                vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
                token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
                strict_load=bool(args.strict_load),
                infer_batch_size=args.infer_batch_size,
            )
    finally:
        cleanup_ddp()
