| model | group | enc | hand IoU | P | R | OB IoU | area pred/hand | bldg pred/hand | best-thr IoU (oracle) |
|---|---|---|---|---|---|---|---|---|---|
| ws_r34_labelqty_c0 | weak-sup OB | resnet34 | 0.632 | 0.741 | 0.810 | 0.636 | 1.09 | 1344/2191 | 0.640 @ 0.3 |
| ws_r34_airsinit | weak-sup OB | resnet34 | 0.605 | 0.770 | 0.738 | 0.654 | 0.96 | 1592/2191 | 0.629 @ 0.3 |
| ws_r34_imagenet | weak-sup OB | resnet34 | 0.605 | 0.770 | 0.737 | 0.655 | 0.96 | 1614/2191 | 0.628 @ 0.3 |
| ws_r34_boundary_relaxed | weak-sup OB | resnet34 | 0.603 | 0.782 | 0.725 | 0.642 | 0.93 | 1445/2191 | 0.621 @ 0.3 |
| st_sep_t050_s43 | OB + self-train | mit_b2 | 0.600 | 0.778 | 0.724 | 0.644 | 0.93 | 1763/2191 | 0.610 @ 0.3 |
| ws_b2_uneroded_s43 | weak-sup OB | mit_b2 | 0.599 | 0.767 | 0.732 | 0.664 | 0.95 | 1619/2191 | 0.623 @ 0.3 |
| ws_b2_uneroded | weak-sup OB | mit_b2 | 0.593 | 0.771 | 0.719 | 0.671 | 0.93 | 1673/2191 | 0.618 @ 0.3 |
| ws_b2_uneroded_b | weak-sup OB | mit_b2 | 0.591 | 0.771 | 0.717 | 0.671 | 0.93 | 1678/2191 | 0.617 @ 0.3 |
| ws_b2_erode02 | weak-sup OB | mit_b2 | 0.582 | 0.777 | 0.698 | 0.664 | 0.90 | 1904/2191 | 0.610 @ 0.3 |
| ws_b2_erode04_s43 | weak-sup OB | mit_b2 | 0.578 | 0.789 | 0.684 | 0.647 | 0.87 | 2128/2191 | 0.610 @ 0.3 |
| ws_b5_erode04 | weak-sup OB | mit_b5 | 0.571 | 0.788 | 0.674 | 0.663 | 0.85 | 2373/2191 | 0.604 @ 0.3 |
| st_sep_t050 | OB + self-train | mit_b2 | 0.570 | 0.791 | 0.671 | 0.646 | 0.85 | 2196/2191 | 0.579 @ 0.3 |
| ws_b2_erode04_s44 | weak-sup OB | mit_b2 | 0.568 | 0.790 | 0.669 | 0.653 | 0.85 | 2306/2191 | 0.602 @ 0.3 |
| ws_b2_erode04_DEFAULT | weak-sup OB | mit_b2 | 0.564 | 0.794 | 0.660 | 0.647 | 0.83 | 2278/2191 | 0.601 @ 0.3 |
| st_sep_pseudo_e02 | OB + self-train | mit_b2 | 0.554 | 0.800 | 0.642 | 0.633 | 0.80 | 2250/2191 | 0.563 @ 0.3 |
| st_sep_pseudo_e04 | OB + self-train | mit_b2 | 0.528 | 0.812 | 0.602 | 0.610 | 0.74 | 2309/2191 | 0.538 @ 0.3 |
| ws_b2_erode08 | weak-sup OB | mit_b2 | 0.526 | 0.818 | 0.596 | 0.592 | 0.73 | 3468/2191 | 0.575 @ 0.3 |
| open_buildings | reference | - | 0.504 | 0.744 | 0.609 | 1.000 | 0.82 | 2701/2191 | 0.504 @ 0.5 |
| st_sep_t080 | OB + self-train | mit_b2 | 0.474 | 0.828 | 0.526 | 0.571 | 0.64 | 3066/2191 | 0.488 @ 0.3 |
| af_R_last | annotation-free | resnet34 | 0.317 | 0.766 | 0.351 | 0.286 | 0.46 | 1776/2191 | 0.379 @ 0.3 |
| af_duda_last | annotation-free | resnet34 | 0.275 | 0.713 | 0.309 | 0.255 | 0.43 | 1167/2191 | 0.290 @ 0.3 |
| airs_seed | airs-only | resnet34 | 0.154 | 0.799 | 0.160 | 0.139 | 0.20 | 942/2191 | 0.174 @ 0.3 |
| af_D_last | annotation-free | resnet34 | 0.130 | 0.771 | 0.136 | 0.119 | 0.18 | 1202/2191 | 0.157 @ 0.3 |
| af_G_last | annotation-free | resnet34 | 0.128 | 0.805 | 0.133 | 0.120 | 0.16 | 943/2191 | 0.156 @ 0.3 |
| af_C2_last | annotation-free | resnet34 | 0.112 | 0.742 | 0.117 | 0.106 | 0.16 | 937/2191 | 0.140 @ 0.3 |
| af_M_last | annotation-free | resnet34 | 0.105 | 0.805 | 0.108 | 0.099 | 0.13 | 918/2191 | 0.128 @ 0.3 |
| af_A_last | annotation-free | resnet34 | 0.100 | 0.835 | 0.102 | 0.097 | 0.12 | 703/2191 | 0.115 @ 0.3 |
| af_B_last | annotation-free | resnet34 | 0.067 | 0.851 | 0.068 | 0.064 | 0.08 | 491/2191 | 0.080 @ 0.3 |
| af_D2_last | annotation-free | resnet34 | 0.061 | 0.733 | 0.063 | 0.061 | 0.09 | 415/2191 | 0.072 @ 0.3 |
| af_C_last | annotation-free | resnet34 | 0.039 | 0.882 | 0.039 | 0.037 | 0.04 | 291/2191 | 0.047 @ 0.3 |
