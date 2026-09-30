# Edge deployment: NVIDIA Jetson benchmark plan

The pipeline already produces what an edge deployment needs: a static-shape ONNX model
(`models/detector.onnx`, opset 17), a model card with the exact input spec, and a benchmark
command that writes comparable JSON results per device. This page is the procedure for running
it on a Jetson (e.g. Orin Nano / Orin NX / AGX Orin) once the hardware is available.

> Status: the harness is tested on x86 CPU. No Jetson numbers are claimed until measured.

## 1. Prepare the device

```bash
# JetPack 6.x ships CUDA, cuDNN and TensorRT
sudo nvpmodel -q                     # note the power mode: results are only comparable per mode
sudo nvpmodel -m 0 && sudo jetson_clocks
pip install onnxruntime-gpu          # use the wheel matching your JetPack (Jetson Zoo / NVIDIA index)
git clone <repo> && cd av-sensor-dataops && pip install -e ".[serve]"
dvc pull models/detector.onnx        # or scp the promoted model
```

## 2. TensorRT engines

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/detector.onnx \
    --saveEngine=models/detector_fp16.engine --fp16 --useSpinWait --iterations=500

# INT8 needs calibration images from the SAME distribution as deployment:
# use data/ml_ready/yolo/images/val (never test) and keep the calibration cache
# under DVC so the engine is reproducible.
/usr/src/tensorrt/bin/trtexec --onnx=models/detector.onnx --int8 --fp16 \
    --calib=models/int8_calib.cache --saveEngine=models/detector_int8.engine
```

## 3. Measure

```bash
avdata benchmark --runs 500 --providers TensorrtExecutionProvider CUDAExecutionProvider
avdata benchmark --runs 500 --providers CUDAExecutionProvider
avdata benchmark --runs 200 --providers CPUExecutionProvider
tegrastats --interval 500 > reports/benchmarks/tegrastats.log &   # power, GPU load, memory
```

Each run writes `reports/benchmarks/<host>_<provider>.json` with p50/p95/p99 for preprocess,
inference, postprocess and total, plus FPS.

## 4. Report template

| Device / power mode | Provider | Precision | p50 total (ms) | p95 total (ms) | FPS | mAP50 (test) | mAP50 (night) |
|---|---|---|---|---|---|---|---|
| laptop CPU | CPU | FP32 | | | | | |
| Jetson … / MAXN | CUDA | FP32 | | | | | |
| Jetson … / MAXN | TensorRT | FP16 | | | | | |
| Jetson … / MAXN | TensorRT | INT8 | | | | | |

Accuracy after quantisation must be re-checked with `avdata evaluate` on the same slices: an INT8
model that is fast but loses the night slice fails the same gate as any other model.

## What to watch on the edge

- Pre/post-processing on the CPU often dominates once inference is on TensorRT; the benchmark splits
  the stages for exactly this reason (move letterboxing to the GPU, e.g. with NVIDIA VPI, if needed).
- Fixed input shape (640×640) keeps TensorRT engines simple; the camera is 1600×900, so letterboxing
  wastes about 30 % of the pixels. A 640×384 export is the next optimisation to measure.
- Thermal throttling: report sustained numbers (after several minutes), not the first second.
