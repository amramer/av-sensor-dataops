"""Declarative schemas (Pandera) for the silver tables.

Schemas catch structural problems: wrong types, nulls, impossible values,
broken keys. Behavioural problems in the sensor data (sync, dropped frames,
weak labels) are handled by the rule-based checks in `checks.py`.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

MODALITIES = ["camera", "lidar", "radar"]

FRAMES = DataFrameSchema(
    {
        "frame_token": Column(str, unique=True),
        "sample_token": Column(str),
        "scene_name": Column(str),
        "channel": Column(str),
        "modality": Column(str, Check.isin(MODALITIES)),
        "timestamp": Column("int64", Check.gt(0)),
        "is_key_frame": Column(bool),
        "filename": Column(str, unique=True),
        "file_size_bytes": Column("int64"),
        "width": Column(int, Check.ge(0)),
        "height": Column(int, Check.ge(0)),
        "ego_speed_mps": Column(float, Check.in_range(0, 70), nullable=True),  # 70 m/s = 252 km/h
    },
    checks=[
        # cameras must declare their image size
        Check(
            lambda df: (df["modality"] != "camera") | ((df["width"] > 0) & (df["height"] > 0)),
            name="camera_has_image_size",
        ),
    ],
    strict=False,
    name="silver.frames",
)

ANNOTATIONS = DataFrameSchema(
    {
        "annotation_token": Column(str, unique=True),
        "sample_token": Column(str),
        "category": Column(str),
        "visibility": Column(int, Check.isin([1, 2, 3, 4])),
        "num_lidar_pts": Column(int, Check.ge(0)),
        "num_radar_pts": Column(int, Check.ge(0)),
        "size": Column(
            object,
            Check(
                lambda s: s.map(lambda v: len(v) == 3 and min(v) > 0),
                element_wise=False,
                name="positive_3d_size",
            ),
        ),
        "rotation": Column(
            object,
            Check(lambda s: s.map(lambda v: len(v) == 4), element_wise=False, name="quaternion"),
        ),
    },
    strict=False,
    name="silver.annotations",
)

FEATURES = DataFrameSchema(
    {
        "frame_token": Column(str, unique=True),
        "decode_ok": Column(bool),
        "brightness": Column(float, Check.in_range(0, 255), nullable=True),
        "n_points": Column(float, Check.ge(0), nullable=True),
    },
    strict=False,
    name="silver.frame_features",
)


def validate(schema: DataFrameSchema, df) -> list[dict]:
    """Validate lazily and return every failure as an issue record."""
    try:
        schema.validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        return [
            {
                "check": "schema",
                "severity": "ERROR",
                "entity_type": schema.name,
                "entity_id": None if row.index != row.index else str(row.index),
                "message": f"{row.column}: {row.check} failed (value={row.failure_case!r})",
            }
            for row in fc.itertuples()
        ]
