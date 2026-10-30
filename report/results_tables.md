### Experiment A: annotation density (random sampling)

| Points per crop | Runs | Best mIoU (mean +/- std) | Mean F1 | Pixel accuracy | Best epoch |
|---:|---:|---|---:|---:|---:|
| 10 | 1 | 0.2048 | 0.3189 | 0.4650 | 10 |
| 50 | 1 | 0.2534 | 0.3764 | 0.5378 | 10 |
| 100 | 1 | 0.2667 | 0.3964 | 0.5430 | 10 |
| 500 | 1 | 0.2486 | 0.3692 | 0.5316 | 10 |
| 1000 | 1 | 0.2502 | 0.3662 | 0.5452 | 10 |

### Experiment B: sampling strategy (100 points per crop)

| Strategy | Runs | Best mIoU (mean +/- std) | Mean F1 | Pixel accuracy |
|---|---:|---|---:|---:|
| class-balanced | 1 | 0.2855 | 0.4395 | 0.4396 |
| random | 1 | 0.2667 | 0.3964 | 0.5430 |

### Experiment C: focal gamma (100 random points)

| gamma | Runs | Best mIoU (mean +/- std) | Mean F1 | Pixel accuracy |
|---:|---:|---|---:|---:|
| 0 (partial CE) | 1 | 0.2456 | 0.3693 | 0.5199 |
| 1 | 1 | 0.2817 | 0.4087 | 0.5602 |
| 2 | 1 | 0.2667 | 0.3964 | 0.5430 |
| 3 | 1 | 0.2509 | 0.3781 | 0.5245 |

### Per-class IoU

| Setting | background | building | road | water | barren | forest | agriculture |
|---|---:|---:|---:|---:|---:|---:|---:|
| class balanced 100pts | 0.281 | 0.307 | 0.303 | 0.410 | 0.206 | 0.310 | 0.181 |
| random 1000pts | 0.468 | 0.150 | 0.034 | 0.425 | 0.000 | 0.277 | 0.396 |
| random 100pts | 0.455 | 0.196 | 0.116 | 0.405 | 0.009 | 0.305 | 0.382 |
| random 10pts | 0.391 | 0.169 | 0.011 | 0.295 | 0.037 | 0.209 | 0.322 |

### Run details

| Run | Strategy | Points | Mean points seen | Epochs | Best epoch | Best mIoU | Wall clock (min) | Validation (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sampling_class_balanced_100pts_seed42 | class_balanced | 100 | 100.0 | 10 | 10 | 0.2855 | 13.0 | 9.9 |
| density_10pts_seed42 | random | 10 | 10.0 | 10 | 10 | 0.2048 | 12.3 | 9.5 |
| density_50pts_seed42 | random | 50 | 50.0 | 10 | 10 | 0.2534 | 11.5 | 9.0 |
| density_100pts_seed42 | random | 100 | 100.0 | 10 | 10 | 0.2667 | 11.6 | 9.1 |
| density_500pts_seed42 | random | 500 | 500.0 | 10 | 10 | 0.2486 | 11.4 | 8.9 |
| density_1000pts_seed42 | random | 1000 | 1000.0 | 10 | 10 | 0.2502 | 11.5 | 9.0 |

