# PicoDet Sweep

The PicoDet sweep uses the 9 general-purpose models in PaddleDetection
`release/2.9` at commit `b25522a0f4bde8c80603f3ba5e3472059972e3b5`:
XS/S/M at 320 and 416, plus L at 320, 416, and 640.

The training task uses the audited two-class COCO contract from the NanoDet
sweep. It keeps anatomical `left_palm` and `right_palm`, disables horizontal
flip, and uses the same train/dev/gate/test boundaries. Each variant runs
serially on GPU 0, validates every five epochs, and stops after eight
validation checks without a `0.001` mAP improvement.

The RTX 5090 requires the validated CUDA 12.9 Paddle nightly recorded in
`scripts/setup_picodet_env.sh`. Stable Paddle `3.3.1/cu126` exposed CUDA but
returned incorrect elementwise GPU results during the mandatory tensor gate;
do not substitute it.

All outputs are ignored under `artifacts/picodet-v001/`. The GPU monitor stops
the sweep after three consecutive samples at or above 82C. The completed
models will be ranked by dev mAP; the top three receive controlled
learning-rate and weight-decay tuning, while all nine receive static batch-1
ONNX export and inference-consistency checks.
