# Speaker notes: slides that changed (Round 2 deck)

Energy numbers are placeholders from the earlier run until the student rerun is done.

## Slide 9: Method overview
This is the whole method on one slide. The AIRS model fails on Jaipur, so we start from CrossEarth, a remote-sensing foundation model already trained on the WHU building dataset. We run it on about 8,900 unlabelled Jaipur tiles and use its outputs in two ways. As hard pseudo-labels, they self-train CrossEarth's own adapters, which gives 0.717. As soft targets, they train a small ResNet-34 student, which gives 0.706 plus or minus 0.005 over three seeds. The student is the model we deploy, because it is small and stable across seeds. Neither route uses a Jaipur label for training. The city-wide energy number comes from running the student over the whole city.

## Slide 15: From pixels to energy
Each roof pixel covers 0.266 by 0.266 metres, so counting pixels gives roof area. The usable share of a roof is the one thing we could not measure: obstacle mapping at this resolution is unreliable on cluttered Indian roofs. So we state an assumption of 50% and show a range from 30 to 60%. Panels convert about 20% of sunlight, which at the standard 1 kW per square metre gives capacity in kWp. Yearly energy is capacity times Jaipur's PVOUT from the Global Solar Atlas, about 1,630 kWh per kWp, times 0.95 for rooftop losses. We do not also multiply by a performance ratio, because PVOUT already includes those losses and doing it twice would underestimate by about 22%.

## Slide 18: What failed
Every method that adapts from the AIRS model stays below about 0.32 on hand-drawn roofs: our self-training arms and our own run of DUDA, a 2026 domain-adaptation method. Changing image colours with FDA or histogram matching made things worse than doing nothing. Much of the problem is calibration: the AIRS model predicts about 6% roof where the city has 28%. We also tried training on Google's Open Buildings footprints after shifting them onto the roofs. That reaches 0.706 in Jaipur, but it needs a footprint database for the region, the footprints match real roofs at only 0.50 IoU, and the shift was tuned on hand labels. So it worked here but is not a portable method. These are our implementations from a small Western source, not the authors' numbers.

## Slide 19: What worked
Two routes worked, using only the CrossEarth foundation model and unlabelled Jaipur tiles. A small ResNet-34 student reaches 0.706, and self-training CrossEarth itself reaches 0.717. Test-time augmentation, averaging eight flips and rotations, adds about 0.01. Bigger encoders, seed ensembles, an edge-weighted loss and fine-tuning on 76 hand crops did not move the score beyond noise. A DINOv3 encoder was slightly worse, and refining the teacher with SAM made it worse in all 12 settings we tried.

## Slide 20: Results vs baselines
On the same 76 test crops, self-training lifts CrossEarth from 0.677 to 0.717. The student, at 0.706, also beats its own teacher by about five times its seed noise. The 0.717 and the 0.706 student are about equal: the gap is inside the confidence interval of about 0.03, and 0.717 is one run. The older baselines were scored on all 152 crops at a fixed 0.5 threshold, so read them as context only. Against supervised Western results we are at about 86% of the Inria best, but that is a different dataset with thousands of labels, so it shows scale, not a head-to-head.

## Slide 21: Qualitative examples
Here are four hand-labelled crops, chosen for variety, not for score. Our self-trained model follows the visible roof outlines. The dense and the cluttered crops are still hard, at 0.48 and 0.32 IoU. Two crops are from the calibration half and two from the test half. Most remaining mistakes are at roof edges and where neighbouring roofs touch, which the next slide measures.

## Slide 23: Jaipur energy estimate
Over the 168 square kilometres we processed, the technical potential at an assumed 50% usable roof is about 4.4 to 5.0 GWp, or 6.8 to 7.7 terawatt-hours a year. The two ends of the range come from the area bias on the calibration and test halves. The usable share is not measured, so we show it as a range: at 30% the capacity is about 2.7 to 3.0 GWp, and at 60% it is 5.3 to 6.0. In our error budget that one assumption dominates, and segmentation is only about 2%. Existing rooftop panels, which our panel model finds, take off less than 0.1 GWp. This is technical potential, not a forecast of what will be installed.

## Slide 24: Our contributions
The building blocks are known; what we add is the setting, the test set and the finding. We found no public hand-labelled roof benchmark for a mid-sized Indian city, so we built one, and we used it only to set thresholds, never to train. We showed that a remote-sensing foundation model can be self-trained on unlabelled Jaipur tiles to 0.717, that its pseudo-label cut-offs must be calibrated, and that a small student keeps almost all of that accuracy. The main lesson is that adapting from a small Western model failed, while a strong foundation-model source plus self-training worked. This is an application and evaluation contribution, not a new algorithm.

## Slide 25: Conclusion
To sum up: the model that did well on New Zealand roofs collapsed on Jaipur, and classical domain adaptation from it did not rescue it. Starting from a remote-sensing foundation model and self-training it on unlabelled Jaipur tiles gave our best score, 0.717, and a small student keeps 0.706. What is left is mostly edge error, which better labels would address more than bigger models. For energy, the potential at an assumed 50% usable roof is about 4.4 to 5.0 GWp, and the usable-roof assumption matters far more than segmentation accuracy.
