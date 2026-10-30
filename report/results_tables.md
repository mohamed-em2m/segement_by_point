### Experiment A: annotation density (random sampling)

| Points per crop | Runs | Best mIoU (mean +/- std) | Mean F1 | Pixel accuracy | Best epoch |
|---:|---:|---|---:|---:|---:|
| 10 | 1 | 0.2362 | 0.3606 | 0.4981 | 10 |

### Experiment B: sampling strategy (100 points per crop)

| Strategy | Runs | Best mIoU (mean +/- std) | Mean F1 | Pixel accuracy |
|---|---:|---|---:|---:|
| class-balanced | 1 | 0.2715 | 0.4225 | 0.4195 |

### Per-class IoU

| Setting | background | building | road | water | barren | forest | agriculture |
|---|---:|---:|---:|---:|---:|---:|---:|
| class balanced 100pts | 0.267 | 0.314 | 0.311 | 0.355 | 0.203 | 0.302 | 0.148 |
| random 10pts | 0.410 | 0.169 | 0.115 | 0.340 | 0.000 | 0.250 | 0.370 |

### Run details

| Run | Strategy | Points | Mean points seen | Epochs | Best epoch | Best mIoU | Wall clock (min) | Validation (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sampling_class_balanced_100pts_seed42 | class_balanced | 100 | 100.0 | 10 | 10 | 0.2715 | 28.7 | 23.4 |
| density_10pts_seed42 | random | 10 | 10.0 | 10 | 10 | 0.2362 | 25.9 | 22.2 |

