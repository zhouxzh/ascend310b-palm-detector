# Pinned NanoDet upstream

| Field | Value |
| --- | --- |
| Repository | `https://github.com/RangiLyu/nanodet.git` |
| Ref | tag `v1.0.0` |
| Immutable commit | `d3fb34fa91d6020f273d6d063bf324dcd97bac12` |
| License | Apache License 2.0 |
| Local acquisition | `./scripts/fetch_nanodet.sh` |

The external checkout lives at `third_party/nanodet/` and is Git-ignored. The
wrapper scripts record its actual commit in each run manifest and refuse to use
the training path when the checkout is absent. No upstream source, model
checkpoint, or unverified release file is silently committed to this repository.

For the baseline, `scripts/acquire_nanodet_pretrained.sh` retrieves the official
COCO checkpoint published at NanoDet release `v1.0.0-alpha-1`:

```text
https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416_checkpoint.ckpt
```

The input size in the source checkpoint name is not the target export shape.
Its compatible backbone/FPN weights initialize the static 320 model; the
80-class source prediction layers are shape-mismatched and are deliberately
reinitialized for the two-class head by NanoDet's checked loader. The script
records bytes and SHA-256 after download in an ignored local manifest used by
the run launcher. The project config sets `backbone.pretrain: false` so NanoDet
does not download an additional implicit Torch Hub ShuffleNet checkpoint.
