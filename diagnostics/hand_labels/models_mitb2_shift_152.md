| model | group | enc | hand IoU | P | R | OB IoU | area pred/hand | bldg pred/hand | best-thr IoU (oracle) |
|---|---|---|---|---|---|---|---|---|---|
| r34_c0_shift | weak-sup OB shifted | resnet34 | 0.675 | 0.773 | 0.842 | 0.591 | 1.09 | 1335/2191 | 0.681 @ 0.3 |
| mitb2_c0_shift | weak-sup OB shifted | mit_b2 | 0.668 | 0.773 | 0.831 | 0.603 | 1.07 | 1344/2191 | 0.677 @ 0.3 |
| ws_b2_uneroded | weak-sup OB | mit_b2 | 0.593 | 0.771 | 0.719 | 0.671 | 0.93 | 1673/2191 | 0.618 @ 0.3 |
| open_buildings | reference | - | 0.504 | 0.744 | 0.609 | 1.000 | 0.82 | 2701/2191 | 0.504 @ 0.5 |
