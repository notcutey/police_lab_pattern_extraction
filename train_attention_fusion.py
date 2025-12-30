#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
import glob
import random
import os.path as osp
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict, Counter
from contextlib import contextmanager
import torchvision.transforms as transforms
import numpy as np
import psutil
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from llm_machine import LLMTextEncoder
from llm_machine import VisionTextSigLIP
from llm_machine.data_linked import ImageDatasetMultiLabel, ImageCollatorMulti, TextCollatorSingle
from networks.RetrievalNet_token import Token
from networks.RetrievalNet_effi import Token as Token_1

from llm_machine import train_step_linked
from llm_machine import log_print


# ===================== #
#   기본 설정 / 경로     #
# ===================== #

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

image_size = (1024, 512)
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD  = [0.26862954, 0.26130258, 0.27577711]

ITEMS_IMG_PATH = "/home/policelab_l40s/llm_prompt/llm_prompt/json_file/shoerinics_group_gt.jsonl"
ITEMS_TXT_PATH = "/home/policelab_l40s/llm_prompt/llm_prompt/json_file/최종_txt_multimodal_train.jsonl"

TOKEN_CKPT_PATH = None

CKPT_DIR = "/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip"
os.makedirs(CKPT_DIR, exist_ok=True)

EPOCHS = 600
BATCH_SIZE_IMG = 64         # ✅ DDP에서는 "per-GPU batch"로 해석한다.
NUM_WORKERS_IMG = 4

LR = 1e-5
WEIGHT_DECAY = 0.01

LABEL_HIT_AT_K = 5
LOG_EVERY_STEPS = 10

DEBUG_BACKPROP = True
DEBUG_SAMPLE_PARAMS = 3
DEBUG_EVERY_STEPS = 1


# ===================== #
#   DDP 최소 조건 유틸   #
# ===================== #

def ddp_is_on() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

def ddp_setup() -> Tuple[int, int, int]:
    """
    torchrun으로 실행될 때:
      - RANK, WORLD_SIZE, LOCAL_RANK 환경변수가 자동으로 세팅된다.
      - 이를 기반으로 NCCL 프로세스 그룹을 초기화한다.
    """
    if not ddp_is_on():
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dist.barrier()
    return rank, world_size, local_rank

def ddp_cleanup():
    if ddp_is_on() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_main_process(rank: int) -> bool:
    return rank == 0

def seed_everything(seed: int, rank: int = 0):
    seed = int(seed) + int(rank) * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


# ===================== #
#     유틸 함수 묶음     #
# ===================== #

def _now():
    return time.strftime("%H:%M:%S")

def _fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def _mem_gb() -> str:
    if torch.cuda.is_available():
        mb = torch.cuda.max_memory_allocated() / (1024**2)
        return f"GPU max {mb:,.0f} MB"
    return f"RAM used {psutil.Process().memory_info().rss / (1024**3):.2f} GB"

def log(msg: str, rank: int = 0):
    if is_main_process(rank):
        print(f"[{_now()}] {msg}", flush=True)

@contextmanager
def timeit(tag: str, rank: int = 0):
    t0 = time.perf_counter()
    log(f"▶ {tag} ...", rank=rank)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log(f"✔ {tag} done in {_fmt_eta(dt)}  ({_mem_gb()})", rank=rank)

def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ===================== #
#   체크포인트 유틸      #
# ===================== #

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

def save_weights(vt, token_model, optimizer, scaler, epoch, global_step, tag="last", save_full_state=True):
    """
    ✅ DDP에서도 단일 GPU에서도 그대로 로드 가능하도록 vt는 unwrap하여 저장한다.
    """
    vt_unwrapped = unwrap_ddp(vt)
    tok_unwrapped = unwrap_ddp(token_model)

    vt_sd = vt_unwrapped.state_dict()
    token_sd = tok_unwrapped.state_dict()

    vt_pth_path = os.path.join(CKPT_DIR, f"vt_{tag}_final_self_attention_multimodal_1_Token_8_resenet.pth")
    token_pth_path = os.path.join(CKPT_DIR, f"token_{tag}_final_self_attention_multimodal_1_Token_8_resnet.pth")

    torch.save({"state_dict": vt_sd}, vt_pth_path)
    torch.save({"state_dict": token_sd}, token_pth_path)
    print(f"[Save] vt (wrapped)   -> {vt_pth_path}")
    print(f"[Save] token (wrapped)-> {token_pth_path}")

    if save_full_state:
        full_path = os.path.join(CKPT_DIR, f"train_state_{tag}.pt")
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "vt": vt_sd,
                "token": token_sd,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
            },
            full_path,
        )
        print(f"[Save] full ckpt -> {full_path}")

def load_vt_from_path(vt: torch.nn.Module, path: str, map_location="cpu", strict: bool = False):
    assert os.path.isfile(path), f"vt ckpt not found: {path}"
    raw = _safe_torch_load(path, map_location=map_location)

    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
    else:
        sd = _unwrap_state_dict(raw)
    sd = _strip_prefix_if_present(sd, "module.")

    vt_unwrapped = unwrap_ddp(vt)
    incompatible = vt_unwrapped.load_state_dict(sd, strict=strict)

    miss = list(getattr(incompatible, "missing_keys", []))
    unexp = list(getattr(incompatible, "unexpected_keys", []))
    if miss:
        print(f"[LoadVT] missing_keys (first 30): {miss[:30]}")
    if unexp:
        print(f"[LoadVT] unexpected_keys (first 30): {unexp[:30]}")

    print(f"[LoadVT] loaded vt weights from {path} (strict={strict})")

def load_token_from_path(
    token_model: torch.nn.Module,
    path: str,
    map_location="cpu",
    strict: bool = False,   # ✅ 호출 호환용(무시해도 됨)
):
    assert os.path.isfile(path), f"token ckpt not found: {path}"

    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    # ✅ B 코드와 동일: ['state_dict']에서 꺼내기
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        # 혹시 .pth가 state_dict 자체로 저장된 케이스까지 커버
        state_dict = ckpt if isinstance(ckpt, dict) else {}
        if len(state_dict) == 0:
            raise KeyError(f"'state_dict' not found and ckpt is not a dict: {path}")

    # module. prefix 제거
    if len(state_dict) > 0 and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    model_wo_ddp = unwrap_ddp(token_model)

    # ✅ B 코드와 동일하게 strict=False 로드 (strict 인자는 무시)
    incompatible = model_wo_ddp.load_state_dict(state_dict, strict=False)

    miss = list(getattr(incompatible, "missing_keys", []))
    unexp = list(getattr(incompatible, "unexpected_keys", []))
    print(f"[LoadToken=FULL] loaded from {path}")
    if miss:
        print(f"  missing keys: {len(miss)} (first 10): {miss[:10]}")
    if unexp:
        print(f"  unexpected keys: {len(unexp)} (first 10): {unexp[:10]}")


def load_train_state_or_pair(vt, token_model, tag: str = "last") -> Optional[int]:
    state_path = os.path.join(CKPT_DIR, f"train_state_{tag}.pt")
    vt_path    = os.path.join(CKPT_DIR, f"vt_{tag}_final_cross_attention_multimodal.pth")
    token_path = os.path.join(CKPT_DIR, f"token_{tag}_final_cross_attention_multimodal.pth")

    vt_u = unwrap_ddp(vt)
    tok_u = unwrap_ddp(token_model)

    epoch = None
    if os.path.isfile(state_path):
        raw = _safe_torch_load(state_path, map_location="cpu")
        if "vt" in raw:
            vt_u.load_state_dict(raw["vt"], strict=False)
        if "token" in raw:
            tok_u.load_state_dict(raw["token"], strict=False)
        epoch = int(raw.get("epoch", 0))
        print(f"[Load] full train_state from {state_path} (epoch={epoch})")
        return epoch

    ok = False
    if os.path.isfile(vt_path):
        raw_vt = _safe_torch_load(vt_path, map_location="cpu")
        vt_sd = _unwrap_state_dict(raw_vt)
        vt_sd = _strip_prefix_if_present(vt_sd, "module.")
        vt_u.load_state_dict(vt_sd, strict=False)
        ok = True
    if os.path.isfile(token_path):
        raw_tok = _safe_torch_load(token_path, map_location="cpu")
        tok_sd = _unwrap_state_dict(raw_tok)
        tok_sd = _strip_prefix_if_present(tok_sd, "module.")
        tok_u.load_state_dict(tok_sd, strict=False)
        ok = True

    if ok:
        print(f"[Load] vt/token from pair files: {vt_path} , {token_path}")
    else:
        print(f"[Load] No checkpoint found for tag='{tag}' under {CKPT_DIR}")
    return epoch

def _load_resume_checkpoint(
    vt,
    token_model,
    optimizer,
    scaler,
    resume_path: Optional[str],
    resume_tag: Optional[str],
    device: str,
    resume_all: bool = False
) -> Tuple[int, int]:
    vt_u = unwrap_ddp(vt)
    tok_u = unwrap_ddp(token_model)

    epoch = 0
    global_step = 0
    if resume_path is not None and len(str(resume_path)) > 0:
        assert os.path.isfile(resume_path), f"resume_path not found: {resume_path}"
        print(f"[Resume] Loading checkpoint from file: {resume_path}")
        raw = _safe_torch_load(resume_path, map_location=device)

        if "vt" in raw:
            vt_u.load_state_dict(raw["vt"], strict=False)
        if "token" in raw:
            tok_u.load_state_dict(raw["token"], strict=False)

        if resume_all:
            if "optimizer" in raw and raw["optimizer"] is not None:
                optimizer.load_state_dict(raw["optimizer"])
            if "scaler" in raw and raw["scaler"] is not None and scaler is not None:
                scaler.load_state_dict(raw["scaler"])

        epoch = int(raw.get("epoch", 0))
        global_step = int(raw.get("global_step", 0))
        print(f"[Resume] Loaded: epoch={epoch}, global_step={global_step}, resume_all={resume_all}")
        return epoch + 1, global_step

    if resume_tag is not None and len(str(resume_tag)) > 0:
        state_path = os.path.join(CKPT_DIR, f"train_state_{resume_tag}.pt")
        if os.path.isfile(state_path):
            print(f"[Resume] Loading checkpoint by tag: {state_path}")
            raw = _safe_torch_load(state_path, map_location=device)

            if "vt" in raw:
                vt_u.load_state_dict(raw["vt"], strict=False)
            if "token" in raw:
                tok_u.load_state_dict(raw["token"], strict=False)

            if resume_all:
                if "optimizer" in raw and raw["optimizer"] is not None:
                    optimizer.load_state_dict(raw["optimizer"])
                if "scaler" in raw and raw["scaler"] is not None and scaler is not None:
                    scaler.load_state_dict(raw["scaler"])

            epoch = int(raw.get("epoch", 0))
            global_step = int(raw.get("global_step", 0))
            print(f"[Resume] Loaded: epoch={epoch}, global_step={global_step}, resume_all={resume_all}")
            return epoch + 1, global_step
        print(f"[Resume] No checkpoint for tag='{resume_tag}' under {CKPT_DIR}")
    return 1, 0


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
#  멀티라벨 메트릭(R@K)  #
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
            raise TypeError("labels 텐서는 [B, C] multi-hot 형태여야 한다.")
        return [set(map(int, labs)) for labs in labels]

    raise TypeError("batch_img에서 라벨 정보를 찾을 수 없다. (label_sets 또는 labels 필요)")

@torch.no_grad()
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
        if (min_score is not None) and (s < min_score):
            break
        lab = int(labels_of_texts[j])
        if lab in seen_labels:
            continue
        seen_labels.add(lab)
        chosen_idx.append(int(j))
        chosen_scores.append(s)
        if len(chosen_idx) >= k:
            break

    return chosen_idx, chosen_scores

@torch.no_grad()
def _compute_label_hit_ratio_at_k(
    vt,
    token_model,
    images_t: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_label_sets: List[set],
    text_label_ids: List[int],
    k: int = 5,
) -> Tuple[float, int, int]:
    device = next(unwrap_ddp(vt).parameters()).device
    use_amp = (device.type == "cuda")
    amp_dtype = torch.float32

    images_t = images_t.to(device, non_blocking=True)
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)

    vt_u = unwrap_ddp(vt)

    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
        feats = token_model.forward_test(images_t)
        out = vt_u(feats, input_ids, attention_mask, targets=None)

        candidate_keys = ["logits", "logits_per_image", "sims", "similarity", "scores"]
        if isinstance(out, dict):
            scores = None
            for k_name in candidate_keys:
                if k_name in out:
                    scores = out[k_name]
                    break
            if scores is None:
                raise RuntimeError(f"VisionTextSigLIP forward 결과에서 score 행렬 키({candidate_keys})를 찾지 못했다.")
        else:
            scores = out

    if scores.dim() != 2:
        raise ValueError(f"scores 텐서는 [B, M] 2D 여야 하는데, shape={scores.shape} 입니다.")

    B, M = scores.shape
    if B == 0 or M == 0:
        return float('nan'), 0, 0

    if len(text_label_ids) != M:
        raise ValueError(f"text_label_ids 길이({len(text_label_ids)})와 scores의 두번째 차원 M({M})이 일치해야 한다.")

    scores_np = scores.detach().cpu().numpy()

    total_hits = 0
    total_gt = 0

    for i in range(B):
        gt = image_label_sets[i]
        if len(gt) == 0:
            continue

        chosen_idx, _ = _topk_unique_by_label(scores_np[i], text_label_ids, k=k, min_score=None)
        pred_labels = {int(text_label_ids[j]) for j in chosen_idx}
        hits = len(pred_labels.intersection(gt))
        total_hits += hits
        total_gt += len(gt)

    ratio = (total_hits / total_gt * 100.0) if total_gt > 0 else float("nan")
    return ratio, total_hits, total_gt


# ===================== #
#   이미지 Dataset 유틸  #
# ===================== #

class ImageFromList(torch.utils.data.Dataset):
    def __init__(self, Image_paths=None, transforms=None, imsize=None, bbox=None, loader=None):
        super(ImageFromList, self).__init__()
        self.Image_paths = Image_paths or []
        self.transforms = transforms
        self.bbox = bbox
        self.imsize = imsize
        self.loader = loader if loader is not None else self.pil_loader
        self.len = len(self.Image_paths)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index):
        path = self.Image_paths[index]
        img = self.loader(path)

        if self.bbox is not None:
            img = img.crop(self.bbox[index])

        if self.imsize is not None:
            if isinstance(self.imsize, int):
                imsize = (self.imsize, self.imsize)
            else:
                imsize = self.imsize
            img = T.Resize(imsize, interpolation=InterpolationMode.BICUBIC)(img)

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return self.len


# ===================== #
#   코사인 거리/유사도    #
# ===================== #

@torch.no_grad()
def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2
    device = a.device
    batch_size = 50

    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)

    sim = torch.empty((a.size(0), b.size(0)), device=device, dtype=a.dtype)

    for i in range(0, a.size(0), batch_size):
        end_i = min(i + batch_size, a.size(0))
        a_chunk = a[i:end_i]
        for j in range(0, b.size(0), batch_size):
            end_j = min(j + batch_size, b.size(0))
            b_chunk = b[j:end_j]
            dot_product = a_chunk @ b_chunk.t()
            sim[i:end_i, j:end_j] = 1.0 - dot_product

    return sim

def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = torch.nn.functional.normalize(A, dim=1)
    B = torch.nn.functional.normalize(B, dim=1)
    return A @ B.t()


# ===================== #
# fused vector 추출 (match에서 사용) #
# ===================== #

@torch.no_grad()
def extract_fused_vectors(
    vt: VisionTextSigLIP,
    token_model: Token,
    token_model_search: Token_1,
    loader: DataLoader,
    t_all: torch.Tensor,     # (M, D)
    weight_img: float,
    weight_txt: float,
    attn_temp: float,
    device: torch.device,
    topk_attn: int = 0,
) -> torch.Tensor:
    """
    - attn이 가장 높은 TOP_N prototype 임베딩을 평균(mean)
    - fused = norm(weight_img*image_feats + weight_txt*text_mean)
    """
    vt_u = unwrap_ddp(vt)
    vt_u.eval()
    token_model.eval()
    token_model_search.eval()  # ✅ 추가: search 모델 eval

    use_amp = (device.type == "cuda")
    try:
        amp_dtype = torch.float32 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float32

    num_images = len(loader.dataset)
    D = t_all.size(1)
    all_fused = torch.zeros(num_images, D, dtype=torch.float32, device=device)

    idx_start = 0
    TOP_N = 5

    for images in tqdm(loader, desc="Extract fused (mean topN prototypes)", total=len(loader)):
        images = images.to(device, non_blocking=True)
        bsz = images.size(0)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            image_feats = token_model.forward_test(images)
            image_feat_final = token_model_search.forward_test(images)

            v = vt_u._project_image_feats(image_feats)
            scores = vt_u._cross_attend_image_to_text(v, t_all)

        attn = torch.softmax(attn_temp * scores, dim=-1)
        top_vals, top_idx = torch.topk(attn, k=TOP_N, dim=-1)

        top_embs = t_all[top_idx]               # (B,TOP_N,D)
        text_mean = top_embs.mean(dim=1)        # (B,D)
        text_mean = F.normalize(text_mean, p=2, dim=1)


        # image_feats_norm = F.normalize(image_feat_final, p=2, dim=1)

        fused = (weight_img * image_feat_final) + (weight_txt * text_mean)
        fused = F.normalize(fused, p=2, dim=1)

        all_fused[idx_start:idx_start + bsz] = fused
        idx_start += bsz

    return all_fused


# ===================== #
# 모델 빌드 함수
# ===================== #

def build_model(
    device: str,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    token_ckpt_search: Optional[str] = None,

    strict_load: bool = False
):
    token_model = Token(outputdim=1024, classifier_num=3821).to(device)
    token_model_search = Token_1(outputdim=1024, classifier_num=3821).to(device)

    for name, p in token_model.named_parameters():
        if name.startswith("backbone"):
            p.requires_grad = True
        else:
            p.requires_grad = True
    token_model.train()

    if token_ckpt_path:
        try:
            load_token_from_path(token_model, token_ckpt_path, map_location="cpu")
        except Exception as e:
            print(f"[Token] Failed to load TOKEN from '{token_ckpt_path}': {e}")
    if token_ckpt_search:
        try:
            load_token_from_path(token_model_search, token_ckpt_search, map_location="cpu", strict=strict_load)
        except Exception as e:
            print(f"[TokenSearch] Failed to load from '{token_ckpt_search}': {e}")


    text_encoder = LLMTextEncoder(
        model_name=MODEL_NAME,
        device=device,
        dtype=torch.bfloat16,
        train_llm=True,
        use_lora=True,
        lora_r=8, lora_alpha=16, lora_dropout=0.1,
        pooling="mean",
    )
    vt = VisionTextSigLIP(
        text_encoder=text_encoder,
        vision_dim=1024,
        proj_out_dim=1024,
        temperature_init=0.06
    ).to(device).train()

    if vt_ckpt_path:
        try:
            load_vt_from_path(vt, vt_ckpt_path, map_location="cpu", strict=strict_load)
        except Exception as e:
            print(f"[VT] Failed to load VT from '{vt_ckpt_path}': {e}")

    return vt, token_model, token_model_search


# ===================== #
# 학습 루프 (DDP 최소 조건 포함)
# ===================== #

def main_train(
    image_size=image_size,
    norm_mean=tuple(NORM_MEAN),
    norm_std=tuple(NORM_STD),
    white_bg_fill=True,
    allow_flip=False,
    save_interval=5,
    resume_tag: Optional[str] = None,
    resume_path: Optional[str] = None,
    resume_all: bool = False,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    token_ckpt_search: Optional[str] = None,
    strict_load: bool = False,
    save_full_state: bool = True,
):
    # ✅ DDP 초기화 (최소조건 1,2)
    rank, world_size, local_rank = ddp_setup()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = (device == "cuda")
    seed_everything(42, rank=rank)

    device_for_model = f"cuda:{local_rank}" if (device == "cuda") else "cpu"

    vt, token_model,token_model_search = build_model(
        device=device_for_model,
        vt_ckpt_path=vt_ckpt_path,
        token_ckpt_path=token_ckpt_path,
        token_ckpt_search=token_ckpt_search,
        strict_load=strict_load
    )

    # ✅ vt를 DDP로 감싼다 (최소조건 4)
    if ddp_is_on():
        vt = DDP(vt, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    vt_u = unwrap_ddp(vt)  # 토크나이저 접근/로드/유틸에 사용

    assert os.path.isfile(ITEMS_IMG_PATH), f"not found: {ITEMS_IMG_PATH}"
    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
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

    # ✅ DistributedSampler 적용 + DataLoader shuffle 끄기 (최소조건 3)
    if ddp_is_on():
        img_sampler = DistributedSampler(
            img_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle_flag = False
    else:
        img_sampler = None
        shuffle_flag = True

    img_loader = DataLoader(
        img_ds,
        batch_size=BATCH_SIZE_IMG,            # per-GPU batch
        shuffle=shuffle_flag,
        sampler=img_sampler,
        num_workers=NUM_WORKERS_IMG,
        collate_fn=ImageCollatorMulti(),
        pin_memory=(device == "cuda"),
        persistent_workers=(NUM_WORKERS_IMG > 0),
        drop_last=False,
    )

    steps_per_epoch = len(img_loader)
    num_images = len(img_ds)

    if is_main_process(rank):
        print(
            f"[Data] #images={num_images} | batch_size(perGPU)={BATCH_SIZE_IMG} | "
            f"world_size={world_size} | global_batch={BATCH_SIZE_IMG*world_size} | steps/epoch={steps_per_epoch}"
        )

    trainable = [p for p in vt_u.parameters() if p.requires_grad] + \
            [p for p in token_model.parameters() if p.requires_grad]



    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    start_epoch, global_step = _load_resume_checkpoint(
        vt=vt,  # DDP wrapper 전달해도 내부에서 unwrap 처리한다.
        token_model=token_model,
        optimizer=optimizer,
        scaler=scaler,
        resume_path=resume_path if resume_path else None,
        resume_tag=resume_tag if resume_tag else None,
        device=device_for_model,
        resume_all=resume_all,
    )

    total_steps = EPOCHS * steps_per_epoch
    ema_iter_time = None
    start_time = time.perf_counter()
    tokenizer = vt_u.text_encoder.tokenizer

    grad_probe = GradProbe(token_model, name_prefix="token")
    if DEBUG_BACKPROP:
        grad_probe.attach()
    snap = ParamSnapshot(token_model, max_params=DEBUG_SAMPLE_PARAMS)

    for epoch in range(start_epoch, EPOCHS + 1):
        # ✅ epoch마다 shard shuffle 동기화 (최소조건 3)
        if ddp_is_on() and hasattr(img_loader.sampler, "set_epoch"):
            img_loader.sampler.set_epoch(epoch)

        if device == "cuda":
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

            # ✅ train_step_linked는 vt가 DDP여도 forward/backward 동작한다.
            logs = train_step_linked(vt, token_model, batch_img, batch_txt, optimizer, scaler)

            cur_loss = float(logs.get("loss", float("nan")))
            cur_temp = float(logs.get("temp", float("nan")))

            if DEBUG_BACKPROP and (global_step % DEBUG_EVERY_STEPS == 0):
                snap.take_after()
                _ = snap.changed_flags()
                _ = grad_probe.summary()

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

                device_t = next(vt_u.parameters()).device
                images_t = images_t.to(device_t, non_blocking=True)
                input_ids = batch_txt.input_ids.to(device_t, non_blocking=True)
                attention_mask = batch_txt.attention_mask.to(device_t, non_blocking=True)

                ratio, hits, gt = _compute_label_hit_ratio_at_k(
                    vt=vt,
                    token_model=token_model,
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

            max_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
            lr_cur = optimizer.param_groups[0]["lr"]

            # ✅ 로그는 rank0만
            if is_main_process(rank) and ((step % LOG_EVERY_STEPS) == 0):
                cum_ratio = (cum_hits / cum_gt * 100.0) if cum_gt > 0 else float("nan")
                print(
                    f">> Train Epoch: [{epoch}] "
                    f"[{step}/{steps_per_epoch}] "
                    f"eta: {format_eta(eta_sec)} "
                    f"VT contrastive loss: {cur_loss:.4f} "
                    f"Label-Hit@{LABEL_HIT_AT_K}: batch {batch_ratio:6.3f}% | epoch {cum_ratio:6.3f}% "
                    f"iter time: {iter_time:.4f} s "
                    f"lr: {lr_cur:.2e} "
                    f"max mem: {int(max_mem_mb)} MB"
                )

        epoch_time = time.perf_counter() - epoch_start
        if is_main_process(rank):
            print(f">> Epoch [{epoch}] done in {format_eta(epoch_time)}")

        # ✅ 저장은 rank0만
        if is_main_process(rank) and (epoch % save_interval == 0):
            save_weights(vt, token_model, optimizer, scaler, epoch, global_step,
                         tag=f"epoch{epoch:03d}", save_full_state=save_full_state)

        # ✅ epoch 끝 barrier(선택이지만 안전)
        if ddp_is_on():
            dist.barrier()

    total_time = time.perf_counter() - start_time
    if is_main_process(rank):
        print(f">> Training done in {format_eta(total_time)}")
        save_weights(vt, token_model, optimizer, scaler, epoch=EPOCHS, global_step=global_step,
                     tag="last", save_full_state=save_full_state)

    ddp_cleanup()


# ===================== #
# 아래: infer/match/cluster는 원코드 흐름 유지
# (DDP 학습 최소조건과 충돌 없게 unwrap만 보강)
# ===================== #

@torch.no_grad()
def precompute_text_embeddings_torch(
    vt: VisionTextSigLIP,
    items_txt,
    device: torch.device,
    text_batch_size: int = 256,
    max_length: int = 128,
):
    vt_u = unwrap_ddp(vt)
    tokenizer = vt_u.text_encoder.tokenizer
    collate_txt = TextCollatorSingle(tokenizer, max_length=max_length)

    label_to_texts = defaultdict(list)
    for it in items_txt:
        lab = int(it["label"])
        txt = it["text"]
        label_to_texts[lab].append(txt)

    selected_items = []
    for lab, txt_list in label_to_texts.items():
        chosen_text = random.choice(txt_list)
        selected_items.append({"text": chosen_text, "label": lab})

    proto_labels = [it["label"] for it in selected_items]
    proto_texts  = [it["text"]  for it in selected_items]

    use_amp = (device.type == "cuda")
    try:
        amp_dtype = torch.float32 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float16

    all_embs = []
    total = len(selected_items)
    s = 0

    while s < total:
        e = min(s + text_batch_size, total)
        batch = selected_items[s:e]

        batch_txt = collate_txt(batch)
        input_ids = batch_txt.input_ids.to(device, non_blocking=True)
        attention_mask = batch_txt.attention_mask.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            t_batch = vt_u.encode_texts(input_ids, attention_mask)

        all_embs.append(t_batch.float())
        s = e

    t_all = torch.cat(all_embs, dim=0)

    print(f"[Text] unique labels = {len(selected_items)}")
    print(f"[Text] precomputed text embeddings: shape={t_all.shape}")
    return t_all, proto_labels, proto_texts

@torch.no_grad()
def apply_text_self_attention(vt: VisionTextSigLIP, t_all: torch.Tensor) -> torch.Tensor:
    vt_u = unwrap_ddp(vt)
    if getattr(vt_u, "use_text_self_attn", False) and (vt_u.text_self_attn_block is not None):
        t_all = vt_u.text_self_attn_block(t_all)
    return t_all

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
            out.append(p)
            seen.add(p)
    return out

def _extract_name_by_regex(p: str) -> str:
    m = re.search(r"/([^/]+)\.", p)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(p))[0]

def _load_image_paths_from_jsonl(path: str) -> List[str]:
    assert os.path.isfile(path), f"jsonl not found: {path}"
    rows = load_jsonl(path)
    out = []
    for r in rows:
        p = r.get("image_path", None)
        if isinstance(p, str) and len(p) > 0:
            out.append(p)
    if len(out) == 0:
        raise RuntimeError(f"no image_path in {path}")
    return out


# ---------------------
# Infer/Match/Cluster (원래 로직 유지, 필요 시 추가로 수정 가능)
# ---------------------

@torch.no_grad()
def run_infer(
    ckpt_tag: str = "last",
    topk: int = 5,
    text_batch: int = 256,
    image_paths: Optional[List[str]] = None,
    max_demo_images: int = 8,
    vt_ckpt_path: Optional[str] = None,
    token_ckpt_path: Optional[str] = None,
    strict_load: bool = False,
    fused_outdir: Optional[str] = None,
    lam: float = 0.2,
    attn_temp: float = 50.0,
    infer_batch_size: int = 64,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vt, token_model,token_model_search = build_model(
        device=device,
        vt_ckpt_path=vt_ckpt_path,
        token_ckpt_path=token_ckpt_path,
        strict_load=strict_load,
    )
    if not vt_ckpt_path and not token_ckpt_path:
        load_train_state_or_pair(vt, token_model, tag=ckpt_tag)

    vt_u = unwrap_ddp(vt)
    vt_u.eval()
    token_model.eval()
    token_model_search.eval()

    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_txt = load_jsonl(ITEMS_TXT_PATH)

    t_all, proto_labels, proto_texts = precompute_text_embeddings_torch(
        vt=vt_u,
        items_txt=items_txt,
        device=torch.device(device),
        text_batch_size=text_batch,
        max_length=128,
    )
    t_all = t_all.to(device, non_blocking=True)
    t_all = apply_text_self_attention(vt_u, t_all)

    if not image_paths:
        assert os.path.isfile(ITEMS_IMG_PATH), f"not found: {ITEMS_IMG_PATH}"
        items_img = load_jsonl(ITEMS_IMG_PATH)
        image_paths = [it["image_path"] for it in items_img][:max_demo_images]

    tfm = T.Compose([
        T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    ds = ImageFromList(Image_paths=image_paths, imsize=image_size, bbox=None, transforms=tfm)
    loader = DataLoader(
        ds,
        batch_size=infer_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS_IMG,
        pin_memory=(device == "cuda"),
    )

    use_amp = (device == "cuda")
    try:
        amp_dtype = torch.float32 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    except Exception:
        amp_dtype = torch.float32

    weight_img = 1.0 - lam
    weight_txt = lam

    if fused_outdir is None or len(str(fused_outdir).strip()) == 0:
        fused_outdir = os.path.join(os.path.dirname(image_paths[0]), "fused_vecs")
    os.makedirs(fused_outdir, exist_ok=True)

    fused_list = []
    meta_all = {}
    global_idx = 0

    for images in tqdm(loader, desc="Infer (batched)"):
        images = images.to(device, non_blocking=True)
        bsz = images.size(0)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            image_feats = token_model.forward_test(images)
            image_feats_final = token_model_search.forward_test(images)

            v = vt_u._project_image_feats(image_feats)
            scores = vt_u._cross_attend_image_to_text(v, t_all)

        attn = torch.softmax(attn_temp * scores, dim=-1)
        h_attn = attn @ t_all
        h_attn = F.normalize(h_attn, p=2, dim=1)

        image_feats_norm = F.normalize(image_feats_final, p=2, dim=1)
        fused = (weight_img * image_feats_norm) + (weight_txt * h_attn)
        fused = F.normalize(fused, p=2, dim=1)

        fused_list.append(fused.detach().cpu().to(torch.float32))
        attn_np = attn.detach().cpu().numpy()

        batch_paths = image_paths[global_idx:global_idx + bsz]
        for i in range(bsz):
            img_path = batch_paths[i]
            attn_i = attn_np[i]
            chosen_idx, _ = _topk_unique_by_label(attn_i, proto_labels, k=topk, min_score=None)

            meta_all[os.path.basename(img_path)] = [
                {"rank": r + 1, "text_index": int(ti), "label": int(proto_labels[ti]),
                 "attn": float(attn_i[ti]), "text": proto_texts[ti]}
                for r, ti in enumerate(chosen_idx)
            ]

        global_idx += bsz

    fused_all = torch.cat(fused_list, dim=0)
    fused_vecs = fused_all.numpy()

    save_obj = {"embeddings": fused_vecs, "image_paths": image_paths}
    npy_path = os.path.join(fused_outdir, "all_images_fused_with_paths.npy")
    np.save(npy_path, save_obj, allow_pickle=True)

    meta_path = os.path.join(fused_outdir, f"all_images_fused_xattn.meta_top{topk}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print(f"[Infer] saved fused embeddings + image paths: {npy_path}")
    print(f"[Infer] saved meta json: {meta_path}")

@torch.no_grad()
def run_match_fused(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vt, token_model,token_model_search = build_model(
        device=device,
        vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
        token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
        token_ckpt_search=args.token_ckpt_search if len(args.token_ckpt_search) > 0 else None,

        strict_load=bool(args.strict_load),
    )
    if (len(args.vt_ckpt_path) == 0) and (len(args.token_ckpt_path) == 0):
        load_train_state_or_pair(vt, token_model, tag=args.ckpt_tag)

    vt_u = unwrap_ddp(vt)
    vt_u.eval()
    token_model.eval()

    assert os.path.isfile(ITEMS_TXT_PATH), f"not found: {ITEMS_TXT_PATH}"
    items_txt = load_jsonl(ITEMS_TXT_PATH)

    t_all, proto_labels, proto_texts = precompute_text_embeddings_torch(
        vt=vt_u,
        items_txt=items_txt,
        device=torch.device(device),
        text_batch_size=args.text_batch,
        max_length=128,
    )
    t_all = t_all.to(device, non_blocking=True)
    t_all = apply_text_self_attention(vt_u, t_all)

    assert len(args.query_jsonl) > 0 and len(args.ref_jsonl) > 0
    query_paths = _load_image_paths_from_jsonl(args.query_jsonl)
    ref_paths   = _load_image_paths_from_jsonl(args.ref_jsonl)

    query_order = [_extract_name_by_regex(p) for p in query_paths]
    ref_names   = [_extract_name_by_regex(p) for p in ref_paths]
    ref_indices = {name: i for i, name in enumerate(ref_names)}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([transforms.ToTensor(), normalize])
    bs = BATCH_SIZE_IMG
    query_loader = DataLoader(ImageFromList(Image_paths=query_paths, imsize=image_size, transforms=tfm),
                              batch_size=bs, shuffle=False, num_workers=NUM_WORKERS_IMG,
                              pin_memory=(device == "cuda"))
    ref_loader = DataLoader(ImageFromList(Image_paths=ref_paths, imsize=image_size, transforms=tfm),
                            batch_size=bs, shuffle=False, num_workers=NUM_WORKERS_IMG,
                            pin_memory=(device == "cuda"))

    lam = args.lam
    weight_img = 1.0 - lam
    weight_txt = lam

    query_vecs = extract_fused_vectors(vt_u, token_model, token_model_search, query_loader, t_all, weight_img, weight_txt,
                                       args.attn_temp, torch.device(device), topk_attn=0)
    ref_vecs = extract_fused_vectors(vt_u, token_model,token_model_search, ref_loader, t_all, weight_img, weight_txt,
                                     args.attn_temp, torch.device(device), topk_attn=0)

    distances = cosine_distance_matrix(query_vecs, ref_vecs)
    sorted_distances, sorted_indices = torch.sort(distances, dim=1)

    log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names, args)
    print("[Match] log_print 완료")

@torch.no_grad()
def run_cluster(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Cluster] device = {device}")

    vt, token_model = build_model(
        device=device,
        vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
        token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
        strict_load=bool(args.strict_load),
    )
    vt_u = unwrap_ddp(vt)
    vt_u.eval()

    assert os.path.isfile(ITEMS_TXT_PATH), f"[Cluster] not found: {ITEMS_TXT_PATH}"
    items_txt = load_jsonl(ITEMS_TXT_PATH)
    print(f"[Cluster] loaded {len(items_txt)} rows from ITEMS_TXT_PATH")

    t_all, proto_labels, proto_texts = precompute_text_embeddings_torch(
        vt=vt_u,
        items_txt=items_txt,
        device=torch.device(device),
        text_batch_size=args.text_batch,
        max_length=128,
    )

    t_all = t_all.to(device, non_blocking=True)
    h_attn = apply_text_self_attention(vt_u, t_all).detach().cpu().to(torch.float32)

    out_npy = args.cluster_out_npy
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    np.save(out_npy, h_attn.numpy())
    print(f"[Cluster] saved h_attn vectors to: {out_npy}")

    out_meta = args.cluster_out_meta
    os.makedirs(os.path.dirname(out_meta), exist_ok=True)
    with open(out_meta, "w", encoding="utf-8") as f:
        for idx, (lab, txt) in enumerate(zip(proto_labels, proto_texts)):
            f.write(json.dumps({"index": int(idx), "label": int(lab), "text": txt}, ensure_ascii=False) + "\n")
    print(f"[Cluster] saved meta jsonl to: {out_meta}  (num_lines={len(proto_labels)})")


# ===================== #
#    엔트리 포인트       #
# ===================== #

def build_argparser():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="train",
                    choices=["train", "infer", "both", "match", "cluster"])
    ap.add_argument("--ckpt_tag", type=str, default="last")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--text_batch", type=int, default=128)
    ap.add_argument("--infer_images", type=str, default="/root/project/llm_prompt/test_crime")
    ap.add_argument("--max_demo_images", type=int, default=1208)

    ap.add_argument("--cluster_out_npy", type=str,
                    default="/root/project/llm_prompt_new/text_cluster/text_h_attn.npy")
    ap.add_argument("--cluster_out_meta", type=str,
                    default="/root/project/llm_prompt_new/text_cluster/text_h_attn_meta.jsonl")

    ap.add_argument("--query_jsonl", type=str,
                    default="/home/policelab_l40s/llm_prompt/llm_prompt/json_file/items_img_1050_filtered.jsonl")
    ap.add_argument("--ref_jsonl", type=str,
                    default="/home/policelab_l40s/llm_prompt/llm_prompt/json_file/items_img_all_rgb_group.jsonl")
    ap.add_argument("--desc_sim", type=int, default=1)
    ap.add_argument("--log_topk", type=int, default=5)

    ap.add_argument("--save", type=str, default="/home/miruware/ieoo0321/Origin_Token/Output")
    ap.add_argument("--testcsv", type=str, default="/home/policelab_l40s/llm_prompt/llm_prompt/label_test_multimodal_1208.csv")

    ap.add_argument("--resume_path", type=str, default="")
    ap.add_argument("--resume_tag", type=str, default="")
    ap.add_argument("--resume_all", type=int, default=0)

    ap.add_argument("--vt_ckpt_path", type=str, default="/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/vt_epoch020_final_self_attention_multimodal_1_Token_8_resenet.pth")
    ap.add_argument("--token_ckpt_path", type=str,
                    default="/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/token_epoch015_final_self_attention_multimodal_1_Token_8_resnet.pth")
    ap.add_argument("--token_ckpt_search", type=str,
                default="")
    ap.add_argument("--strict_load", type=int, default=0)
    ap.add_argument("--save_full_state", type=int, default=1)

    ap.add_argument("--infer_batch_size", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0)
    ap.add_argument("--attn_temp", type=float, default=1)
    ap.add_argument("--topk_attn", type=int, default=4)

    return ap

if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.mode in ("train", "both"):
        resume_path = args.resume_path if len(args.resume_path) > 0 else None
        resume_tag = args.resume_tag if len(args.resume_tag) > 0 else None

        main_train(
            resume_path=resume_path,
            resume_tag=resume_tag,
            resume_all=bool(args.resume_all),
            vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
            token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
            strict_load=bool(args.strict_load),
            save_full_state=bool(args.save_full_state),
        )

    if args.mode in ("infer", "both"):
        paths = _gather_paths(args.infer_images)
        run_infer(
            ckpt_tag=args.ckpt_tag,
            topk=args.topk,
            text_batch=args.text_batch,
            image_paths=paths if len(paths) > 0 else None,
            max_demo_images=args.max_demo_images,
            vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
            token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
            strict_load=bool(args.strict_load),
            fused_outdir=None,
            lam=args.lam,
            attn_temp=args.attn_temp,
            infer_batch_size=args.infer_batch_size,
        )

    if args.mode == "match":
        run_match_fused(args)

    if args.mode == "cluster":
        run_cluster(args)
