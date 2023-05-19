# MuTILs: explainable, multiresolution panoptic segmentation of the breast tumor microenvironment

## Paper
Access [here](https://www.medrxiv.org/content/10.1101/2022.01.08.22268814v1.full.pdf). Citation:
```
Amgad M, Salgado R, Cooper LA. MuTILs: explainable, multiresolution computational scoring 
of Tumor-Infiltrating Lymphocytes in breast carcinomas using clinical guidelines. 
MedRxiv. 2022 Jan 13:2022-01.
```

## Abstract
Tumor-Infiltrating Lymphocytes (TILs) have strong prognostic and predictive value in breast cancer, but their visual assessment is subjective. To improve reproducibility, the International Immuno-oncology Working Group recently released recommendations for the computational assessment of TILs that build on visual scoring guidelines. However, existing resources do not adequately address these recommendations due to the lack of annotation datasets that enable joint, panoptic segmentation of tissue regions and cells. Moreover, existing deep-learning architectures focus entirely on either tissue segmentation or object detection, which complicates the process of TILs assessment by necessitating the use of multiple models with inconsistent predictions. We introduce PanopTILs, a region and cell-level annotation dataset containing 814,886 nuclei from 151 patients, openly accessible at: sites.google.com/view/panoptils. PanopTILs enabled us to develop MuTILs, a convolutional neural network architecture optimized for assessing TILs in accordance with clinical recommendations. MuTILs is a concept bottleneck model designed to be interpretable and to encourage sensible predictions at multiple resolutions. Using a rigorous internal-external cross-validation procedure, MuTILs achieves an AUROC of 0.93 for lymphocyte detection and a DICE coefficient of 0.81 for tumor-associated stroma segmentation. Our computational score closely matched visual scores (Spearman R=0.58, p<0.001). Moreover, our TILs scores had a higher prognostic value than visual scoring, independent of TNM stage and patient age. In conclusion, we introduce a comprehensive open data resource and a novel modeling approach for detailed mapping of the breast tumor microenvironment. 

## Architecture
![image](https://github.com/PathologyDataScience/MuTILs_Panoptic/assets/22067552/e9453cf3-5c9a-4fc3-b12e-8404a27ab48c)

## Sample results
![image](https://github.com/PathologyDataScience/MuTILs_Panoptic/assets/22067552/0e43d964-f560-4e51-b268-de93255ec1bf)

![image](https://github.com/PathologyDataScience/MuTILs_Panoptic/assets/22067552/c3c36f0c-95de-446a-8a9b-3aba172304ce)
