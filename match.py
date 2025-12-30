#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, random
import numpy as np
from typing import List, Optional, Dict, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from PIL import Image

# ==== project modules (프로젝트에 이미 존재한다고 가정) ====
from llm_machine import LLMTextEncoder, VisionTextSigLIP, log_print
from llm_machine.data_linked import TextCollatorSingle

# Token 이름 충돌 방지: Train/Search 분리
from networks import SOLAR as TokenTrainNet
from networks.RetrievalNet_another import SOLAR as TokenSearchNet


# =========================
# (0) 기본 유틸
# =========================

def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def _extract_name_by_regex(p: str) -> str:
    m = re.search(r"/([^/]+)\.", p)
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(p))[0]

def _load_image_paths_from_jsonl(path: str) -> List[str]:
    rows = load_jsonl(path)
    out = []
    for r in rows:
        p = r.get("image_path", None)
        if isinstance(p, str) and len(p) > 0:
            out.append(p)
    if len(out) == 0:
        raise RuntimeError(f"no image_path in {path}")
    return out

def parse_img_size(img_size_arg: str):
    """
    args.img_size:
      - "1024,512"  -> (1024, 512)
      - "1024 512"  -> (1024, 512)
      - "1024x512"  -> (1024, 512)
      - "1024"      -> 1024 (정사각)
    """
    s = str(img_size_arg).strip().lower()
    if "," in s:
        h, w = s.split(",", 1)
        return (int(h.strip()), int(w.strip()))
    if "x" in s:
        h, w = s.split("x", 1)
        return (int(h.strip()), int(w.strip()))
    if " " in s:
        parts = [p for p in s.split() if p]
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    # fallback: int -> square
    return int(s)


# =========================
# (1) Image Dataset (match 전용)
# =========================

def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class ImageFromList(torch.utils.data.Dataset):
    def __init__(self, Image_paths=None, transforms=None, imsize=None, bbox=None, loader=pil_loader):
        super().__init__()
        self.Image_paths = Image_paths or []
        self.transforms = transforms
        self.imsize = imsize
        self.bbox = bbox
        self.loader = loader
        self.len = len(self.Image_paths)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        path = self.Image_paths[idx]
        img = self.loader(path)
        if self.bbox is not None:
            img = img.crop(self.bbox[idx])
        if self.imsize is not None:
            # int면 정사각, tuple이면 그대로
            if isinstance(self.imsize, int):
                size = (self.imsize, self.imsize)
            else:
                size = self.imsize  # (H, W)
            img = transforms.Resize(size)(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img


# =========================
# (2) Cache util (npy)
# =========================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_npy(path: str, arr: np.ndarray):
    _ensure_dir(os.path.dirname(path))
    np.save(path, arr)
    print(f"[Cache] saved: {path}  shape={arr.shape}  dtype={arr.dtype}")

def load_npy(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    print(f"[Cache] loaded: {path}  shape={arr.shape}  dtype={arr.dtype}")
    return arr

def _cache_paths(cache_dir: str, kind: str):
    return {
        "img":  os.path.join(cache_dir, f"{kind}_img_vecs.npy"),
        "txt":  os.path.join(cache_dir, f"{kind}_txt_vecs.npy"),
        "meta": os.path.join(cache_dir, f"{kind}_paths.npy"),
    }

def _safe_cache_root(save_root: str, img_size_tag: str) -> str:
    """
    save_root가 read-only면 HOME 하위로 자동 fallback.
    또한 img_size가 바뀌면 캐시 충돌을 막기 위해 하위 폴더로 분리한다.
    """
    save_root = os.path.abspath(save_root)
    try:
        test_dir = os.path.join(save_root, ".write_test")
        os.makedirs(test_dir, exist_ok=True)
        tmp = os.path.join(test_dir, "tmp.txt")
        with open(tmp, "w") as f:
            f.write("ok")
        os.remove(tmp)
        os.rmdir(test_dir)
        return os.path.join(save_root, "cache_vecs", img_size_tag)
    except Exception:
        fallback = os.path.join(os.path.expanduser("~"), "vec_cache", "cache_vecs", img_size_tag)
        print(f"[Warn] save_root is not writable: {save_root}")
        print(f"[Warn] fallback cache_dir -> {fallback}")
        return fallback


# =========================
# (3) Cosine distance
# =========================

@torch.no_grad()
def cosine_distance_matrix(a: torch.Tensor, b: torch.Tensor, chunk: int = 128) -> torch.Tensor:
    """
    cosine distance = 1 - cos_sim
    a: (N, D), b: (M, D)
    """
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    dist = torch.empty((a.size(0), b.size(0)), device=a.device, dtype=a.dtype)
    for i in range(0, a.size(0), chunk):
        ai = a[i:i+chunk]
        for j in range(0, b.size(0), chunk):
            bj = b[j:j+chunk]
            dist[i:i+ai.size(0), j:j+bj.size(0)] = 1.0 - (ai @ bj.t())
    return dist


# =========================
# (4) Text prototype embedding (라벨당 1개)
# =========================

@torch.no_grad()
def precompute_text_embeddings_one_per_label(
    vt: VisionTextSigLIP,
    items_txt: List[Dict[str, Any]],
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
        label_to_texts[lab].append(it["text"])

    selected = []
    proto_labels = []
    proto_texts = []
    for lab, txts in label_to_texts.items():
        chosen = random.choice(txts)
        selected.append({"text": chosen, "label": lab})
        proto_labels.append(int(lab))
        proto_texts.append(chosen)

    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16

    embs = []
    for s in range(0, len(selected), text_batch_size):
        batch = selected[s:s+text_batch_size]
        bt = collate_txt(batch)
        input_ids = bt.input_ids.to(device, non_blocking=True)
        attention_mask = bt.attention_mask.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            t = vt_u.encode_texts(input_ids, attention_mask)   # (B, D)
        embs.append(t.float())

    t_all = torch.cat(embs, dim=0)  # (M, D)
    print(f"[TextProto] M={t_all.size(0)}  D={t_all.size(1)}")
    return t_all, proto_labels, proto_texts


@torch.no_grad()
def apply_text_self_attention_if_exists(vt: VisionTextSigLIP, t_all: torch.Tensor) -> torch.Tensor:
    vt_u = unwrap_ddp(vt)
    if getattr(vt_u, "use_text_self_attn", False) and (getattr(vt_u, "text_self_attn_block", None) is not None):
        t_all = vt_u.text_self_attn_block(t_all)
    return t_all


# =========================
# (5) Cache extractors
# =========================

@torch.no_grad()
def extract_image_vectors_token_search(
    token_model_search,
    loader: DataLoader,
    device: torch.device,
    total: int,
) -> torch.Tensor:
    token_model_search.eval()
    D = getattr(token_model_search, "outputdim", 1024)
    out = torch.zeros(total, D, dtype=torch.float32, device="cpu")

    idx = 0
    for batch in tqdm(loader, desc="Extract IMG vecs (token_search)", total=len(loader)):
        batch = batch.to(device, non_blocking=True)
        feats = token_model_search.forward_test(batch)      # (B, D)
        feats = F.normalize(feats, p=2, dim=1).detach().cpu().float()
        bsz = feats.size(0)
        out[idx:idx+bsz] = feats
        idx += bsz

    assert idx == total, f"extracted {idx} != total {total}"
    return out


@torch.no_grad()
def extract_text_vectors_xattn_topN(
    vt: VisionTextSigLIP,
    token_model,
    loader: DataLoader,
    t_all: torch.Tensor,
    attn_temp: float,
    topN: int,
    device: torch.device,
    total: int,
) -> torch.Tensor:
    vt_u = unwrap_ddp(vt)
    vt_u.eval()
    token_model.eval()

    t_all = t_all.to(device, non_blocking=True)
    D = t_all.size(1)

    out = torch.zeros(total, D, dtype=torch.float32, device="cpu")

    use_amp = (device.type == "cuda")
    amp_dtype = torch.float16

    idx = 0
    for images in tqdm(loader, desc=f"Extract TXT vecs (xattn top{topN})", total=len(loader)):
        images = images.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            image_feats = token_model.forward_test(images)          # (B, 1024)
            v = vt_u._project_image_feats(image_feats)              # (B, 1024)
            scores = vt_u._cross_attend_image_to_text(v, t_all)     # (B, M)

        attn = torch.softmax(float(attn_temp) * scores, dim=-1)     # (B, M)
        _, top_idx = torch.topk(attn, k=topN, dim=-1)               # (B, topN)
        top_embs = t_all[top_idx]                                   # (B, topN, D)
        txt_mean = top_embs.mean(dim=1)                              # (B, D)
        txt_mean = F.normalize(txt_mean, p=2, dim=1).detach().cpu().float()

        bsz = txt_mean.size(0)
        out[idx:idx+bsz] = txt_mean
        idx += bsz

    assert idx == total, f"extracted {idx} != total {total}"
    return out


# =========================
# (6) Fuse + match
# =========================

@torch.no_grad()
def fuse_and_match_from_cached(
    query_img_vecs: torch.Tensor,   # CPU (Nq, D)
    query_txt_vecs: torch.Tensor,   # CPU (Nq, D)
    ref_img_vecs: torch.Tensor,     # CPU (Nr, D)
    ref_txt_vecs: torch.Tensor,     # CPU (Nr, D)
    lam: float,
    device: torch.device,
):
    w_img = 1.0 - float(lam)
    w_txt = float(lam)

    q = F.normalize(w_img * query_img_vecs.to(device) + w_txt * query_txt_vecs.to(device), p=2, dim=1)
    r = F.normalize(w_img * ref_img_vecs.to(device)   + w_txt * ref_txt_vecs.to(device),   p=2, dim=1)

    distances = cosine_distance_matrix(q, r, chunk=128)
    sorted_distances, sorted_indices = torch.sort(distances, dim=1)
    return sorted_indices, sorted_distances


# =========================
# (7) Model build (match 전용)
# =========================

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def _try_build_token(cls, device: str):
    """
    Token 구현체 ctor가 다를 수 있어, 가능한 조합을 순차 시도한다.
    """
    for kwargs in (
        {"outputdim": 1024, "classifier_num": 3821},
        {"mode": "infer"},
        {},
    ):
        try:
            m = cls(**kwargs).to(device)
            return m
        except TypeError:
            continue
    return cls().to(device)

def build_model_for_match(
    device: str,
    vt_ckpt_path: Optional[str],
    token_ckpt_path: Optional[str],
    token_ckpt_search: Optional[str],
    strict_load: bool,
    load_vt_from_path,
    load_token_from_path,
):
    token_model = _try_build_token(TokenTrainNet, device)
    token_model_search = _try_build_token(TokenSearchNet, device)

    if token_ckpt_path:
        load_token_from_path(token_model, token_ckpt_path, map_location="cpu")
    if token_ckpt_search:
        load_token_from_path(token_model_search, token_ckpt_search, map_location="cpu", strict=strict_load)
    text_encoder = LLMTextEncoder(
    model_name=MODEL_NAME,
    device=device,
    dtype=torch.bfloat16,
    train_llm=True,     # ✅ 학습 당시처럼
    use_lora=True,      # ✅ 학습 당시처럼
    lora_r=8, lora_alpha=16, lora_dropout=0.1,  # ✅ 학습 당시처럼(같은 값)
    pooling="mean",
)


    vt = VisionTextSigLIP(
        text_encoder=text_encoder,
        vision_dim=1024,
        proj_out_dim=1024,
        temperature_init=0.06
    ).to(device)

    if vt_ckpt_path:
        load_vt_from_path(vt, vt_ckpt_path, map_location="cpu", strict=strict_load)

    vt.eval()
    token_model.eval()
    token_model_search.eval()
    return vt, token_model, token_model_search


# =========================
# (8) RUN: match 캐시 버전
# =========================

@torch.no_grad()
def run_match_fused_cached(
    args,
    ITEMS_TXT_PATH: str,
    load_vt_from_path,
    load_token_from_path,
):
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # 1) 모델 로드
    vt, token_model, token_model_search = build_model_for_match(
        device=device_str,
        vt_ckpt_path=args.vt_ckpt_path if len(args.vt_ckpt_path) > 0 else None,
        token_ckpt_path=args.token_ckpt_path if len(args.token_ckpt_path) > 0 else None,
        token_ckpt_search=args.token_ckpt_search if len(args.token_ckpt_search) > 0 else None,
        strict_load=bool(args.strict_load),
        load_vt_from_path=load_vt_from_path,
        load_token_from_path=load_token_from_path,
    )
    vt_u = unwrap_ddp(vt)

    # 2) text prototype
    items_txt = load_jsonl(ITEMS_TXT_PATH)
    t_all, proto_labels, proto_texts = precompute_text_embeddings_one_per_label(
        vt=vt_u,
        items_txt=items_txt,
        device=device,
        text_batch_size=int(args.text_batch),
        max_length=128,
    )
    t_all = apply_text_self_attention_if_exists(vt_u, t_all).to(device, non_blocking=True)

    # 3) query/ref paths
    query_paths = _load_image_paths_from_jsonl(args.query_jsonl)
    ref_paths   = _load_image_paths_from_jsonl(args.ref_jsonl)

    query_order = [_extract_name_by_regex(p) for p in query_paths]
    ref_names   = [_extract_name_by_regex(p) for p in ref_paths]
    ref_indices = {name: i for i, name in enumerate(ref_names)}

    # 4) dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tfm = transforms.Compose([transforms.ToTensor(), normalize])

    bs = int(args.infer_batch_size)

    # ✅ 여기 핵심: "1024,512" -> (1024,512) 로 파싱
    img_size = parse_img_size(args.img_size)
    if isinstance(img_size, int):
        img_size_tag = f"{img_size}x{img_size}"
    else:
        img_size_tag = f"{img_size[0]}x{img_size[1]}"
    print(f"[Match] img_size = {img_size}  (tag={img_size_tag})")

    query_loader = DataLoader(
        ImageFromList(Image_paths=query_paths, imsize=img_size, transforms=tfm),
        batch_size=bs, shuffle=False, num_workers=int(args.num_workers),
        pin_memory=(device_str == "cuda")
    )
    ref_loader = DataLoader(
        ImageFromList(Image_paths=ref_paths, imsize=img_size, transforms=tfm),
        batch_size=bs, shuffle=False, num_workers=int(args.num_workers),
        pin_memory=(device_str == "cuda")
    )

    # 5) cache dir (img_size별로 분리)
    cache_root = _safe_cache_root(args.save, img_size_tag)
    _ensure_dir(cache_root)
    q_cache = _cache_paths(cache_root, "query")
    r_cache = _cache_paths(cache_root, "ref")

    # 6) IMG vecs cache
    if not os.path.exists(q_cache["img"]):
        q_img = extract_image_vectors_token_search(
            token_model_search=token_model_search,
            loader=query_loader,
            device=device,
            total=len(query_paths),
        ).numpy()
        save_npy(q_cache["img"], q_img)
        save_npy(q_cache["meta"], np.array(query_paths, dtype=object))
    else:
        q_img = load_npy(q_cache["img"])

    if not os.path.exists(r_cache["img"]):
        r_img = extract_image_vectors_token_search(
            token_model_search=token_model_search,
            loader=ref_loader,
            device=device,
            total=len(ref_paths),
        ).numpy()
        save_npy(r_cache["img"], r_img)
        save_npy(r_cache["meta"], np.array(ref_paths, dtype=object))
    else:
        r_img = load_npy(r_cache["img"])

    # 7) TXT vecs cache (xattn topN mean)
    topN = int(args.topk_attn)

    if not os.path.exists(q_cache["txt"]):
        q_txt = extract_text_vectors_xattn_topN(
            vt=vt_u,
            token_model=token_model,
            loader=query_loader,
            t_all=t_all,
            attn_temp=float(args.attn_temp),
            topN=topN,
            device=device,
            total=len(query_paths),
        ).numpy()
        save_npy(q_cache["txt"], q_txt)
    else:
        q_txt = load_npy(q_cache["txt"])

    if not os.path.exists(r_cache["txt"]):
        r_txt = extract_text_vectors_xattn_topN(
            vt=vt_u,
            token_model=token_model,
            loader=ref_loader,
            t_all=t_all,
            attn_temp=float(args.attn_temp),
            topN=topN,
            device=device,
            total=len(ref_paths),
        ).numpy()
        save_npy(r_cache["txt"], r_txt)
    else:
        r_txt = load_npy(r_cache["txt"])

    # 8) fuse + match
    q_img_t = torch.from_numpy(q_img).float()
    q_txt_t = torch.from_numpy(q_txt).float()
    r_img_t = torch.from_numpy(r_img).float()
    r_txt_t = torch.from_numpy(r_txt).float()

    sorted_indices, sorted_distances = fuse_and_match_from_cached(
        query_img_vecs=q_img_t,
        query_txt_vecs=q_txt_t,
        ref_img_vecs=r_img_t,
        ref_txt_vecs=r_txt_t,
        lam=float(args.lam),
        device=device,
    )

    # 9) log_print
    log_print(sorted_indices, sorted_distances, query_order, ref_indices, ref_names, args)
    print(f"[Match] done. lam={args.lam}  cache_dir={cache_root}")


# =========================
# (9) args
# =========================

def build_argparser():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", type=str, default="match", choices=["match"])
    ap.add_argument("--query_jsonl", type=str, required=True)
    ap.add_argument("--ref_jsonl", type=str, required=True)

    ap.add_argument("--save", type=str, default="./outputs")  # ✅ 기본 writable
    ap.add_argument("--img_size", type=str, default="1024,512")  # ✅ 핵심: (H,W) 형태
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--infer_batch_size", type=int, default=50)

    ap.add_argument("--text_batch", type=int, default=256)
    ap.add_argument("--attn_temp", type=float, default=1.0)
    ap.add_argument("--topk_attn", type=int, default=5)
    ap.add_argument("--lam", type=float, default=0.0)

    ap.add_argument("--vt_ckpt_path", type=str,
                    default="/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/vt_epoch010_final_self_attention_multimodal_1_SOLAR_all_freeze.pth")
    ap.add_argument("--token_ckpt_path", type=str,
                    default="/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_siglip/token_epoch010_final_self_attention_multimodal_1_SOLAR_all_freeze.pth")
    ap.add_argument("--token_ckpt_search", type=str, default="/home/policelab_l40s/R101-SOLAR_2_new.pth")
    ap.add_argument("--strict_load", type=int, default=0)

    # log_print가 참조하는 필드들(프로젝트 코드에 맞춰 유지)
    ap.add_argument("--log_topk", type=int, default=5)
    ap.add_argument("--testcsv", type=str, default="/home/policelab_l40s/llm_prompt/llm_prompt/label_test_multimodal_1208.csv")

    return ap


# =========================
# (10) main
# =========================

if __name__ == "__main__":
    args = build_argparser().parse_args()

    # ⚠️ 아래 import 경로는 "네 프로젝트에서 load_vt_from_path / load_token_from_path가 정의된 파일"로 바꿔야 한다.
    from train_images_multi_text_single_attention import load_vt_from_path, load_token_from_path

    ITEMS_TXT_PATH = "/home/policelab_l40s/llm_prompt/llm_prompt/json_file/최종_txt_multimodal_train.jsonl"

    run_match_fused_cached(
        args=args,
        ITEMS_TXT_PATH=ITEMS_TXT_PATH,
        load_vt_from_path=load_vt_from_path,
        load_token_from_path=load_token_from_path,
    )
