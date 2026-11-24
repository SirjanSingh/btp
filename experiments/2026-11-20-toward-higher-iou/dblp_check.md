# Step 0: dblp check of the cited papers

dblp blocks automated requests, so this is checked by hand in a real browser.
For each paper, open the search link, find the matching record, and paste the dblp
record URL (the `https://dblp.org/rec/...` link from the record's export menu) into the table.
Mark `not on dblp` if no record exists. arXiv-only papers are listed on dblp under `CoRR`.

| # | Paper (as cited) | dblp search | dblp record URL | Venue on dblp matches? |
|---|---|---|---|---|
| 1 | Ronneberger, Fischer, Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015 | https://dblp.org/search?q=U-Net+Convolutional+Networks+for+Biomedical+Image+Segmentation | https://dblp.org/rec/conf/miccai/RonnebergerFB15 | Yes. MICCAI (3) 2015, pp. 234-241. (CoRR preprint also listed: https://dblp.org/rec/journals/corr/RonnebergerFB15) |
| 2 | He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016 | https://dblp.org/search?q=Deep+Residual+Learning+for+Image+Recognition | https://dblp.org/rec/conf/cvpr/HeZRS16 | Yes. CVPR 2016, pp. 770-778. (CoRR preprint: https://dblp.org/rec/journals/corr/HeZRS15) |
| 3 | Sirko et al. Continental-Scale Building Detection from High Resolution Satellite Imagery. arXiv 2107.12283 (2021) | https://dblp.org/search?q=Continental-Scale+Building+Detection+from+High+Resolution+Satellite+Imagery | https://dblp.org/rec/journals/corr/abs-2107-12283 | Yes. CoRR abs/2107.12283 (2021). arXiv-only on dblp, no journal/conference version listed. |
| 4 | Kaiser, Wegner, Lucchi, Jaggi, Hofmann, Schindler. Learning Aerial Image Segmentation From Online Maps. IEEE TGRS 55(11), 2017 | https://dblp.org/search?q=Learning+Aerial+Image+Segmentation+From+Online+Maps | https://dblp.org/rec/journals/tgrs/KaiserWLJHS17 | Yes. IEEE Trans. Geosci. Remote. Sens. 55(11): 6054-6068 (2017). (CoRR preprint: https://dblp.org/rec/journals/corr/KaiserWLJHS17) |
| 5 | Mnih, Hinton. Learning to Label Aerial Images from Noisy Data. ICML 2012 | https://dblp.org/search?q=Learning+to+Label+Aerial+Images+from+Noisy+Data | https://dblp.org/rec/conf/icml/MnihH12 | Yes. ICML 2012. |
| 6 | Girard, Charpiat, Tarabalka. Noisy Supervision for Correcting Misaligned Cadaster Maps Without Perfect Ground Truth Data. IGARSS 2019 | https://dblp.org/search?q=Noisy+Supervision+for+Correcting+Misaligned+Cadaster+Maps | https://dblp.org/rec/conf/igarss/GirardCT19 | Yes. Record key is conf/igarss/GirardCT19, i.e. IGARSS 2019. (CoRR preprint: https://dblp.org/rec/journals/corr/abs-1903-06529) |
| 7 | Loshchilov, Hutter. Decoupled Weight Decay Regularization. ICLR 2019 | https://dblp.org/search?q=Decoupled+Weight+Decay+Regularization | https://dblp.org/rec/conf/iclr/LoshchilovH19 | Yes. ICLR 2019 (conf/iclr/LoshchilovH19). |
| 8 | Xie et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021 | https://dblp.org/search?q=SegFormer+Simple+and+Efficient+Design | https://dblp.org/rec/conf/nips/XieWYAAL21 | Yes. NeurIPS 2021 (listed as NeurIPS 2021: 12077-12090). (CoRR preprint: https://dblp.org/rec/journals/corr/abs-2105-15203) |
| 9 | Kang et al. DUDA (WACV 2026), arXiv 2504.09814 | https://dblp.org/search?q=2504.09814 | https://dblp.org/rec/journals/corr/abs-2504-09814 | NO. dblp lists only CoRR abs/2504.09814 (2025), no WACV 2026 record (as of 2026-11-20). Cite as arXiv 2025, or confirm WACV 2026 acceptance from the paper/WACV site. |
| 10 | Kasmi et al. 2025, arXiv 2408.07828 (base paper for the UDA comparison) | https://dblp.org/search?q=2408.07828 | https://dblp.org/rec/journals/corr/abs-2408-07828 | NO (year). dblp: CoRR abs/2408.07828 (2024), not 2025. Title on dblp: 'Space-scale Exploration of the Poor Reliability of Deep Learning Models: the Case of the Remote Sensing of Rooftop Photovoltaic Systems'; last author Blanc. No published venue listed. |

Notes for whoever fills this in:
- If a title search finds nothing, search by first author + year.
- If dblp lists a different venue or year than the "as cited" column, write what dblp says;
  the citation in the write-up gets corrected to match.
