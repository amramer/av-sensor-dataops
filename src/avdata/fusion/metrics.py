"""3D detection metrics in the style of the nuScenes detection benchmark.

Matching uses the centre distance on the ground plane (BEV), not IoU, as in
nuScenes. For each class and distance threshold (0.5, 1, 2, 4 m):
predictions are sorted by score and greedily matched to the closest unmatched
ground-truth box of the same sample; AP is the area under the precision /
recall curve above 10 % recall and 10 % precision, normalised to [0, 1].

True-positive errors are measured at the 2 m threshold:
  ATE  translation error (m, BEV centre distance)
  ASE  1 - 3D IoU after aligning centre and orientation (size error)
  AOE  orientation error (rad), taken modulo pi here because the baseline
       cannot tell front from back

NDS-lite = (5 * mAP + sum(1 - min(1, err)) for ATE, ASE, AOE) / 8.
The official NDS also includes velocity and attribute errors, which this
single-frame, attribute-free baseline does not predict.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

MIN_RECALL = 0.1
MIN_PRECISION = 0.1


def _yaw_diff_pi(a: float, b: float) -> float:
    d = abs((a - b) % np.pi)
    return float(min(d, np.pi - d))


def _aligned_iou(s1, s2) -> float:
    inter = np.prod(np.minimum(s1, s2))
    return float(inter / (np.prod(s1) + np.prod(s2) - inter))


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tp[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    prec = ctp / np.maximum(ctp + cfp, 1)
    rec = ctp / n_gt
    rec_grid = np.linspace(0, 1, 101)
    prec_interp = np.interp(rec_grid, rec, prec, right=0)
    # make precision monotonically decreasing (standard interpolation)
    prec_interp = np.maximum.accumulate(prec_interp[::-1])[::-1]
    prec_interp[rec_grid > rec.max() + 1e-9] = 0
    p = prec_interp[round(100 * MIN_RECALL) + 1 :] - MIN_PRECISION
    p[p < 0] = 0
    return float(np.mean(p) / (1 - MIN_PRECISION))


def match(preds: list[dict], gts: list[dict], thr: float) -> tuple[np.ndarray, np.ndarray, list]:
    """Greedy centre-distance matching. Boxes: dicts with sample, cls, center, size, yaw, score."""
    by_sample = defaultdict(list)
    for i, g in enumerate(gts):
        by_sample[g["sample"]].append(i)
    taken: set[int] = set()
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
    tp = np.zeros(len(preds))
    pairs = []
    for i in order:
        p = preds[i]
        best, best_d = None, thr
        for j in by_sample.get(p["sample"], []):
            if j in taken:
                continue
            d = float(
                np.hypot(p["center"][0] - gts[j]["center"][0], p["center"][1] - gts[j]["center"][1])
            )
            if d < best_d:
                best, best_d = j, d
        if best is not None:
            taken.add(best)
            tp[i] = 1
            pairs.append((i, best, best_d))
    scores = np.array([p["score"] for p in preds])
    return tp, scores, pairs


def evaluate(
    preds: list[dict], gts: list[dict], thresholds=(0.5, 1.0, 2.0, 4.0), tp_threshold: float = 2.0
) -> dict:
    classes = sorted({g["cls"] for g in gts})
    per_class: dict[str, dict] = {}
    all_err = {"ate": [], "ase": [], "aoe": []}
    for cls in classes:
        P = [p for p in preds if p["cls"] == cls]
        G = [g for g in gts if g["cls"] == cls]
        aps = {}
        for thr in thresholds:
            tp, scores, pairs = match(P, G, thr)
            aps[f"{thr:g}m"] = average_precision(tp, scores, len(G))
            if thr == tp_threshold:
                ate = [d for _, _, d in pairs]
                ase = [1 - _aligned_iou(P[i]["size"], G[j]["size"]) for i, j, _ in pairs]
                aoe = [_yaw_diff_pi(P[i]["yaw"], G[j]["yaw"]) for i, j, _ in pairs]
                recall = len(pairs) / len(G) if G else float("nan")
        per_class[cls] = {
            "n_gt": len(G),
            "n_pred": len(P),
            "ap": round(float(np.mean(list(aps.values()))), 4),
            "ap_by_threshold": {k: round(v, 4) for k, v in aps.items()},
            f"recall@{tp_threshold:g}m": round(recall, 4),
            "ate": round(float(np.mean(ate)), 4) if ate else None,
            "ase": round(float(np.mean(ase)), 4) if ase else None,
            "aoe": round(float(np.mean(aoe)), 4) if aoe else None,
        }
        all_err["ate"] += ate
        all_err["ase"] += ase
        all_err["aoe"] += aoe

    def cls_mean(key):
        vals = [c[key] for c in per_class.values() if c[key] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    m_ap = cls_mean("ap") or 0.0
    errs = {k: cls_mean(k) for k in ("ate", "ase", "aoe")}
    nds = (5 * m_ap + sum(1 - min(1.0, v if v is not None else 1.0) for v in errs.values())) / 8
    return {
        "classes": classes,
        "mAP": m_ap,
        "mATE": errs["ate"],
        "mASE": errs["ase"],
        "mAOE": errs["aoe"],
        "NDS_lite": round(float(nds), 4),
        "n_gt": len(gts),
        "n_pred": len(preds),
        "per_class": per_class,
    }


def recall_by(preds: list[dict], gts: list[dict], key: str, thr: float = 2.0) -> dict:
    """Recall and ATE per group of ground-truth boxes (e.g. distance bin), class-aware."""
    out = {}
    for group in sorted({g[key] for g in gts}):
        G = [g for g in gts if g[key] == group]
        hits, ate = 0, []
        for cls in {g["cls"] for g in G}:
            _, _, pairs = match(
                [p for p in preds if p["cls"] == cls], [g for g in G if g["cls"] == cls], thr
            )
            hits += len(pairs)
            ate += [d for _, _, d in pairs]
        out[group] = {
            "n_gt": len(G),
            "recall": round(hits / len(G), 4),
            "ate": round(float(np.mean(ate)), 4) if ate else None,
        }
    return out
