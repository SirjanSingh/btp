| model | group | enc | hand IoU | P | R | OB IoU | area pred/hand | bldg pred/hand | best-thr IoU (oracle) |
|---|---|---|---|---|---|---|---|---|---|
| r34_c0_shift | weak-sup OB shifted | resnet34 | 0.675 | 0.773 | 0.842 | 0.591 | 1.09 | 1335/2191 | 0.681 @ 0.3 |
| r34_c0_shift_s43 | weak-sup OB shifted | resnet34 | 0.671 | 0.769 | 0.840 | 0.596 | 1.09 | 1399/2191 | 0.676 @ 0.3 |
| r34_c0_shift_s44 | weak-sup OB shifted | resnet34 | 0.668 | 0.771 | 0.834 | 0.597 | 1.08 | 1349/2191 | 0.675 @ 0.3 |
| ws_r34_labelqty_c0 | weak-sup OB | resnet34 | 0.632 | 0.741 | 0.810 | 0.636 | 1.09 | 1346/2191 | 0.640 @ 0.3 |
| open_buildings | reference | - | 0.504 | 0.744 | 0.609 | 1.000 | 0.82 | 2701/2191 | 0.504 @ 0.5 |
