# Presenter notes, slides 9 to 17 (about 7 minutes)

Timing: about 6.5 minutes at normal speed. Slide 11 (hidden) is optional; skip it to save 20 seconds.
Check before presenting: the slide 9 crop count (152 scored vs 187 labelled), and which model produced the city energy number.

## Slide 9: Datasets (40 s)
We use two datasets. AIRS is from Christchurch, New Zealand: 857 training and 94 validation images at 7.5 cm per pixel, with full roof masks. Jaipur is 16 big tiles, about 175 square kilometres, at 26.6 cm per pixel, so 3.5 times coarser, with no training labels. To test honestly we hand-drew roofs on 152 crops of 512 by 512 pixels and split them by tile into 76 for calibration and 76 for testing. These crops are never used for training.

## Slide 10: Method overview (1 min)
Our AIRS model scores only 0.154 on Jaipur, so we cannot use it directly. Instead we use CrossEarth, a foundation model already trained on the WHU building dataset, as a teacher. We run it on 8,920 Jaipur tiles with no labels and save its roof probability for every pixel: these are soft maps. Then we train a small ResNet-34 U-Net student on those maps, starting from our AIRS weights. This is knowledge distillation. The student scores 0.706 over three seeds, and we use it to measure roofs over the city. No Jaipur label is used for training.
If asked: self-training CrossEarth itself gave 0.717, one run, within the margin of error and much heavier, so it is not our final model.

## Slide 11: What is CrossEarth, hidden (20 s)
CrossEarth is a published model, Gong et al., arXiv 2410.22629. It has a large DINOv2-L vision transformer, small Reins adapter layers inside it, and a Mask2Former head that outputs masks. The authors trained it on WHU buildings; we use their weights unchanged and only run it.

## Slide 12: Teacher, soft map, student (1 min 15 s)
Here is one real tile. The teacher runs on it with test-time augmentation: flipped and rotated copies, averaged. It gives a roof probability from 0 to 1 per pixel, which we keep as is, with no thresholds. The student is a U-Net with a ResNet-34 encoder, starting from AIRS weights. Its target is the teacher's probability itself. The loss is half soft cross-entropy and half soft Dice. We train with AdamW at learning rate 1e-4, a one-cycle schedule, batch size 16, for 6 epochs, about 3,300 steps. Augmentation is flips, rotations and small colour changes. We report the last epoch, never the best one on test.

## Slide 13: Why the teacher's scores need care (30 s)
CrossEarth's scores are squeezed into a narrow band, about 0.27 to 0.73, so the usual 0.5 cut-off cannot be trusted. We pick the student's threshold, about 0.55, on the calibration crops only. The student learns from the soft values directly, so it needs no hard cut-offs.

## Slide 14: From pixels to energy (1 min)
Once we have a roof map, the rest is simple arithmetic in four steps.
One, area: each pixel is 0.266 by 0.266 metres, so roof area is pixel count times that.
Two, usable roof: panels cannot go on water tanks, stairwells or clutter. We could not measure this on Indian roofs, so we assume half the roof is usable, and show a range of 30 to 60 percent.
Three, capacity: India's renewable energy ministry, MNRE, says about 10 square metres of clear roof is needed for 1 kilowatt of panels. So usable area divided by 10 gives kilowatts.
Four, energy: the Global Solar Atlas, a public sunlight map, says each kilowatt of panels in Jaipur makes about 1,630 units of electricity, kilowatt-hours, per year. We multiply by 0.95 for small losses like dust and heat.
Example: 1,000 roof pixels is about 71 square metres, half usable is 35, which is 3.5 kilowatts, and about 5,500 units a year. Panels already on roofs are subtracted.

## Slide 15: Test protocol (35 s)
Earlier, every Jaipur score compared the model with Open Buildings footprints, which were also its training labels, so it measured agreement, not accuracy. Now we use 152 hand-drawn crops from all 16 tiles by 3 annotators, split by tile. The calibration half only picks the threshold; the test half only reports the score, with a margin of error of about plus or minus 0.03. Some older baselines were scored on all 152 crops at a fixed 0.5 threshold, and we mark those separately.

## Slide 16: What failed (50 s)
Our AIRS model alone gives 0.154. Every method that adapts from it stays under 0.32, including about ten self-training variants and our own run of DUDA, a 2026 method, at 0.274. Changing image colours made things worse. The AIRS model predicts about 6 percent roof where the true share is 28. Training on Google's Open Buildings footprints, shifted onto the roofs, reaches 0.706, but it needs a footprint database, the footprints match real roofs at only 0.50 IoU, and the shift was tuned on hand labels, so it is not portable. Refining the teacher with SAM was worse in all 12 settings.

## Slide 17: What worked (50 s)
Two routes worked. Route 1, our final model, is the student at 0.706 plus or minus 0.005, and it beats its own teacher at 0.677. Route 2 self-trains CrossEarth itself: only the adapters and the head are trained, with the backbone frozen, for 6,000 iterations, batch size 2, learning rate 2e-5, about 65 minutes on one V100. It reaches 0.717 from a single run, which is about equal, so we choose the small, stable student. Test-time augmentation adds about 0.01. Seed ensembles, a bigger encoder, an edge-weighted loss and fine-tuning on 76 hand crops did not help.
