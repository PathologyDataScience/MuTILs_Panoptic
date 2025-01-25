# MuTILs: explainable, multiresolution panoptic segmentation of the breast tumor microenvironment

## Paper
Access [here](https://www.nature.com/articles/s41523-024-00663-1). Citation:
```
Shangke Liu, Mohamed Amgad, Deeptej More, Muhammad A. Rathore, Roberto Salgado, Lee A. D. Cooper: A panoptic segmentation dataset and deep-learning approach for explainable scoring of tumor-infiltrating lymphocytes
npj Breast Cancer 10, 52 (2024). https://doi.org/10.1038/s41523-024-00663-1
```

## Abstract
Tumor-Infiltrating Lymphocytes (TILs) have strong prognostic and predictive value in breast cancer, but their visual assessment is subjective. To improve reproducibility, the International Immuno-oncology Working Group recently released recommendations for the computational assessment of TILs that build on visual scoring guidelines. However, existing resources do not adequately address these recommendations due to the lack of annotation datasets that enable joint, panoptic segmentation of tissue regions and cells. Moreover, existing deep-learning methods focus entirely on either tissue segmentation or cell nuclei detection, which complicates the process of TILs assessment by necessitating the use of multiple models and reconciling inconsistent predictions. We introduce PanopTILs, a region and cell-level annotation dataset containing 814,886 nuclei from 151 patients, openly accessible at: [sites.google.com/view/panoptils](https://sites.google.com/view/panoptils/home). Using PanopTILs we developed MuTILs, a neural network optimized for assessing TILs in accordance with clinical recommendations. MuTILs is a concept bottleneck model designed to be interpretable and to encourage sensible predictions at multiple resolutions. Using a rigorous internal-external cross-validation procedure, MuTILs achieves an AUROC of 0.93 for lymphocyte detection and a DICE coefficient of 0.81 for tumor-associated stroma segmentation. Our computational score closely matched visual scores from 2 pathologists (Spearman R = 0.58–0.61, p < 0.001). Moreover, computational TILs scores had a higher prognostic value than visual scores, independent of TNM stage and patient age. In conclusion, we introduce a comprehensive open data resource and a modeling approach for detailed mapping of the breast tumor microenvironment.

## Architecture
![image](https://github.com/PathologyDataScience/MuTILs_Panoptic/assets/22067552/e9453cf3-5c9a-4fc3-b12e-8404a27ab48c)

## Sample results
![image](https://github.com/PathologyDataScience/MuTILs_Panoptic/assets/22067552/0e43d964-f560-4e51-b268-de93255ec1bf)

![image](https://github.com/PathologyDataScience/MuTILs_Panoptic/assets/22067552/c3c36f0c-95de-446a-8a9b-3aba172304ce)

## Usage

### Containerized approach

Use a containerized environment to run and train MuTILs.

MuTILs has been developed as a part of cTME (Computational pipelines for analysis of Tumor MicroEnvironment) project for which there is a publicly available Docker image. For convenience, follow the steps below to set up the environment and make an inference with MuTILs.

Clone this repository then pull the container on your GPU server.

1. `git clone https://github.com/szolgyen/MuTILs_Panoptic`
2. `cd MuTILs_Panoptic`
3. `git switch dev-szolgyen-refaq`
4. `git submodule update --init --recursive`
5. `cd histolab`
6. `git switch dev-szolgyen`
7. `docker pull szolgyen/mutils:v0`

Todo: This is going to be replaced with an installer.

The code is built on Python 3.10.12 and the container environment hosts a 12.0.0 CUDA on a 22.04 Ubuntu system.

### Set configurations

Modify both files of
 - `MuTILs_Panoptic/docker-compose.yaml`
 - `MuTILs_Panoptic/configs/MuTILsWSIRunConfigs.yaml`

 with the correct paths of files and folders on your system.

### Start the container and build Cython modules within the container

8. `docker-compose up`

In a separate terminal, attach to the container:

9. `docker attach MutilsDev1`

Once within the container, set up Cython modules:

10. `cd /home/MuTILs_Panoptic/utils/CythonUtils`
11. `python setup.py build_ext --inplace`
12. `cd /home/`

Todo: This has to be handled by an installer too.

### Run MuTILsWSIRunner.py

Within the container, run the MuTILsWSIRunner.py module to perform inference on your set of slides at the location defined in the configuration YAML file.

13. `python MuTILs_Panoptic/mutils_panoptic/MuTILsWSIRunner.py`

Note: Do not forget to give permission to your folders to make them accessible for MuTILs.

### Model weights

https://huggingface.co/mutils-panoptic/mutils/tree/main
