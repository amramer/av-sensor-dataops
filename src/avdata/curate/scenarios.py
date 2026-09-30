"""Scenario tagging: which operational situations does each sample cover?

Tags come from three sources:
  scene metadata   time of day, weather, location
  ego motion       stopped / slow / moving
  annotations      pedestrians, bicycles, crowded scenes, ...

The same tags drive the coverage KPIs, the regression set and the
per-scenario model evaluation, so "what data do we have" and "where is the
model weak" are answered in the same vocabulary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from avdata.config import Params


def scene_tags(description: str) -> dict[str, str]:
    d = description.lower()
    return {
        "time_of_day": "night" if "night" in d else "day",
        "weather": "rain" if "rain" in d else "clear",
    }


def speed_bin(speed: float, bins: list[float]) -> str:
    if speed != speed:  # NaN
        return "unknown"
    stopped, slow = bins
    if speed < stopped:
        return "stopped"
    return "slow" if speed < slow else "moving"


def tag_samples(frames: pd.DataFrame, annotations: pd.DataFrame, p: Params) -> pd.DataFrame:
    """One row per keyframe sample with scenario tags."""
    ref = frames[frames["is_key_frame"] & (frames["channel"] == p.sensors.reference_channel)]
    s = ref[
        [
            "sample_token",
            "scene_name",
            "scene_description",
            "location",
            "sample_timestamp",
            "ego_speed_mps",
        ]
    ].copy()
    # first frame of a scene has no speed: take the scene's next known value
    s = s.sort_values(["scene_name", "sample_timestamp"])
    s["ego_speed_mps"] = s.groupby("scene_name")["ego_speed_mps"].bfill()

    tags = s["scene_description"].apply(scene_tags).apply(pd.Series)
    s = pd.concat([s, tags], axis=1)
    s["speed_bin"] = [speed_bin(v, p.curation.speed_bins_mps) for v in s["ego_speed_mps"]]

    det = annotations[annotations["det_class"].notna()]
    counts = det.pivot_table(
        index="sample_token",
        columns="det_class",
        values="annotation_token",
        aggfunc="count",
        fill_value=0,
    )
    for cls in p.curation.class_names:
        n = s["sample_token"].map(counts[cls]) if cls in counts else 0
        s[f"n_{cls}"] = pd.Series(n, index=s.index).fillna(0).astype(int)
        s[f"has_{cls}"] = s[f"n_{cls}"] > 0
    s["crowded"] = (
        s["n_pedestrian"] >= p.curation.crowded_min_pedestrians if "n_pedestrian" in s else False
    )

    # flat tag list, e.g. ["night", "rain", "has_bicycle"] - used for regression/slices
    def _tags(r: pd.Series) -> list[str]:
        out = [r["time_of_day"], r["weather"], r["speed_bin"], r["location"]]
        out += [c for c in s.columns if c.startswith("has_") and r[c]]
        if r["crowded"]:
            out.append("crowded")
        return out

    s["tags"] = s.apply(_tags, axis=1)
    return s.reset_index(drop=True)


def assign_splits(samples: pd.DataFrame, p: Params) -> pd.Series:
    """Deterministic scene-level split, stratified by `curation.stratify_by`.

    Scenes of each stratum are shuffled with the seed and interleaved
    round-robin, then cut into test / val / train. This keeps every stratum
    represented in each split where possible and guarantees no scene (and
    therefore no recording) appears in two splits.
    """
    c = p.curation
    rng = np.random.default_rng(c.seed)
    scene_stratum = samples.groupby("scene_name")[c.stratify_by].agg(
        lambda v: v.value_counts().index[0]
    )
    groups = []
    for _, scenes in sorted(scene_stratum.groupby(scene_stratum)):
        names = sorted(scenes.index)
        rng.shuffle(names)
        groups.append(names)
    groups.sort(key=len, reverse=True)
    order: list[str] = []
    for i in range(max(len(g) for g in groups)):
        order += [g[i] for g in groups if i < len(g)]

    n = len(order)
    n_test = max(1, round(n * c.test_fraction)) if n >= 3 and c.test_fraction > 0 else 0
    n_val = max(1, round(n * c.val_fraction)) if n >= 3 and c.val_fraction > 0 else 0
    split = {s: "test" for s in order[:n_test]}
    split |= {s: "val" for s in order[n_test : n_test + n_val]}
    split |= {s: "train" for s in order[n_test + n_val :]}
    return samples["scene_name"].map(split)
