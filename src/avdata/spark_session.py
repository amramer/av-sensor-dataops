"""One place to configure Spark.

Local by default (`local[*]`). To run the same jobs on a cluster, set
SPARK_MASTER (e.g. spark://host:7077, k8s://https://..., yarn) and ship
the package with `spark.submit.pyFiles` or a container image.
"""

from __future__ import annotations

import os
import sys
from functools import cache


@cache
def get_spark(app_name: str = "avdata"):
    from pyspark.sql import SparkSession

    # executors must use the same interpreter (and therefore the avdata package)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
