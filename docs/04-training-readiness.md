# Training Readiness

The current detector targets MediaPipe-style palm regions and predicts the
person's anatomical left or right hand. It trains from the portable HaGRIDv2
derived release at `data/portable-hagridv2-mediapipe-hand/coco-handed-v001`.

| Requirement | Current evidence |
| --- | --- |
| Class and image contract | `configs/nanodet-plus-m-320-portable-hagridv2-handed.yml`; category 1/2 is left/right palm region, input 320 x 320 |
| Split audit | `reports/portable-hagridv2-handed-data-audit-v001.json`: 6,997 train / 825 val / 1,600 test images; hashes and subject/session separation pass |
| Pretrained model | `data/manifests/nanodet-plus-m_416_checkpoint.json` pins the upstream NanoDet checkpoint and its SHA-256 |
| Training environment | `palm-nanodet` and `scripts/run_with_host_nvidia.sh` run NanoDet on the local CUDA GPU |
| Previous two-class candidate | `artifacts/runs/nanodet-plus-m-320-portable-handed__data-v001__seed-42__20261113-1524`: 150 epochs, best checkpoint and ONNX export available |
| Surface-view labels | `coco-handed-surface-v001` contains automatic palm/back/unknown candidates; browser reviews are recorded separately |

The palm detector's `left_palm` and `right_palm` classes do not say whether the
visible surface is the palm or back. The view candidates need documented human
review before being used as ground truth for a surface classifier or as a gate
for palmprint recognition. The portable release is generated from MediaPipe
outputs, so independent target-camera evaluation is also needed.

Atlas 310B ATC conversion, ACL numerical agreement, on-device quality, and
latency remain separate deployment checks. An ONNX export from the training
server does not complete those checks.
