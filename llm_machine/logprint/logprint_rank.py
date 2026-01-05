import os
import math
import numpy as np
from tqdm import tqdm

import logging
from time import strftime, localtime

import pandas as pd
import ast


def get_logger(log_dir: str = "./logs", name: str = "log_print"):
    """
    간단한 파일 로거 생성
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        time_stamp = strftime("%m-%d_%H-%M", localtime())
        log_path = os.path.join(log_dir, f"log_{time_stamp}R_152-delg_2.log")

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)

        logger.addHandler(fh)

    return logger


def log_print(sorted_indices,
              sorted_distances,
              query_order,
              ref_indices,
              ref_names,
              args):

    logger = get_logger(log_dir="./logs", name="log_print")

    # ====== 출력용 top-k 리스트 ======
    TOP_LIST = 500  # top-20만 출력

    # ====== 평가 기준: Top-50 + Top-1%/5%/10% ======
    N_ref = len(ref_names)
    top50_k = 50
    top1p_k = max(1, int(math.ceil(N_ref * 0.01)))
    top5p_k = max(1, int(math.ceil(N_ref * 0.05)))
    top10p_k = max(1, int(math.ceil(N_ref * 0.10)))

    # (혹시 reference가 50보다 적으면 안전하게 클램프)
    top50_k = min(top50_k, N_ref)

    # 카운터
    count_top50 = 0
    count_top1p = 0
    count_top5p = 0
    count_top10p = 0
    total = 0

    rank_rows = []
    df = pd.read_csv("/home/policelab_l40s/llm_prompt/llm_prompt/label_test_814.csv")

    logger.debug(f"sorted_indices.shape: {sorted_indices.shape}")
    logger.debug(f"sorted_distances.shape: {sorted_distances.shape}")
    logger.debug(f"N_ref={N_ref} | top50={top50_k}, top1%={top1p_k}, top5%={top5p_k}, top10%={top10p_k}")

    # ✅ 안전한 index->name 변환 함수
    def idx_to_name(idx: int) -> str:
        try:
            if 0 <= idx < len(ref_names):
                return str(ref_names[idx])
        except Exception:
            pass
        return f"IDX_OUT_OF_RANGE:{idx}"

    for qidx, (top_ranks, top_distances, query_file_name) in tqdm(
        enumerate(zip(sorted_indices, sorted_distances, query_order)),
        total=len(query_order)
    ):
        # ✅ top-20 reference "이름" 추출
        top20_ref_indices = top_ranks[:TOP_LIST].detach().cpu().tolist()
        top20_ref_names = [idx_to_name(int(rid)) for rid in top20_ref_indices]

        top20_ref_str = "[" + ", ".join(
            [f"{i+1}:{name}" for i, name in enumerate(top20_ref_names)]
        ) + "]"

        # GT 찾기
        try:
            gt_file_name = df[df["cropped"] == query_file_name]["gt"].iloc[0]
        except IndexError:
            logger.debug(f"[WARN] GT not found for query: {query_file_name} | top{TOP_LIST}: {top20_ref_str}")
            rank_rows.append({
                "test_name": query_file_name,
                "best_rank": -1,
                "top20_ref_names": top20_ref_str,
                "hit_top50": 0,
                "hit_top1p": 0,
                "hit_top5p": 0,
                "hit_top10p": 0,
            })
            continue

        # GT 라벨 파싱
        try:
            gt_labels = ast.literal_eval(gt_file_name)
            if isinstance(gt_labels, str):
                gt_labels = [gt_labels]
        except Exception:
            gt_labels = [gt_file_name]

        # GT ref index로 변환
        gt_indices = []
        for gt_label in gt_labels:
            if gt_label in ref_indices:
                gt_indices.append(ref_indices[gt_label])
            else:
                logger.debug(f"[WARN] GT label '{gt_label}' not in ref_indices for query '{query_file_name}'")

        # best_rank 계산 (1-based)
        best_rank = float("inf")
        if len(gt_indices) > 0:
            for gt_index in gt_indices:
                involve_index = (top_ranks == gt_index).nonzero(as_tuple=True)[0]
                if len(involve_index) > 0:
                    rank1 = involve_index[0].item() + 1
                    if rank1 < best_rank:
                        best_rank = rank1

        total += 1

        # hit 계산
        if best_rank == float("inf"):
            score = -1
            hit_top50 = 0
            hit_top1p = 0
            hit_top5p = 0
            hit_top10p = 0
        else:
            score = int(best_rank)
            hit_top50 = 1 if score <= top50_k else 0
            hit_top1p = 1 if score <= top1p_k else 0
            hit_top5p = 1 if score <= top5p_k else 0
            hit_top10p = 1 if score <= top10p_k else 0

            count_top50 += hit_top50
            count_top1p += hit_top1p
            count_top5p += hit_top5p
            count_top10p += hit_top10p

        # CSV 저장
        rank_rows.append({
            "test_name": query_file_name,
            "best_rank": score,
            "top20_ref_names": top20_ref_str,
            "hit_top50": hit_top50,
            "hit_top1p": hit_top1p,
            "hit_top5p": hit_top5p,
            "hit_top10p": hit_top10p,
        })

        # 로그 출력
        logger.debug(
            f"query: {query_file_name}, best_rank: {score} | "
            f"Top50({top50_k}):{hit_top50}, Top1%({top1p_k}):{hit_top1p}, "
            f"Top5%({top5p_k}):{hit_top5p}, Top10%({top10p_k}):{hit_top10p} | "
            f"top{TOP_LIST}: {top20_ref_str}"
        )

    # 결과 로그
    if total > 0:
        top50_acc = (count_top50 / total) * 100
        top1p_acc = (count_top1p / total) * 100
        top5p_acc = (count_top5p / total) * 100
        top10p_acc = (count_top10p / total) * 100

        logger.debug(f"FINAL Top-50 accuracy  (k={top50_k})  : {top50_acc} %")
        logger.debug(f"FINAL Top-1% accuracy  (k={top1p_k})  : {top1p_acc} %")
        logger.debug(f"FINAL Top-5% accuracy  (k={top5p_k})  : {top5p_acc} %")
        logger.debug(f"FINAL Top-10% accuracy (k={top10p_k}) : {top10p_acc} %")

    csv_path = "./rank_result.csv"
    pd.DataFrame(rank_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved GT rank CSV to {csv_path}")

    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]

    # 반환: Top-50, Top-1%, Top-5%, Top-10%
    return [
        (count_top50 / total) * 100,
        (count_top1p / total) * 100,
        (count_top5p / total) * 100,
        (count_top10p / total) * 100,
    ]
