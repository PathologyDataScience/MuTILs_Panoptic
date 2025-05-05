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

We recommend using the <i>szolgyen/mutils:v2</i> image from Docker Hub to perform inference with MuTILs. This image is based on Ubuntu 22.04 and includes a Python 3.10.12 virtual environment preconfigured with all the necessary packages for MuTILs. It is built on [nvidia/cuda:12.0.0-base-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.0.0-base-ubuntu22.04/images/sha256-3b6f49136ec6725b6fcc0fc04f2f7711d3d1e22d0328da2ca73e52dbd37fa4b1), providing CUDA 12.0.0 compatibility.

For the list of dependencies and additional details, refer to the [Dockerfile](https://github.com/szolgyen/MuTILs_Panoptic/blob/main/Dockerfile) in this repository.

#### Pull the docker image on your GPU server:

1. `docker pull szolgyen/mutils:v2`

We recommend using a `run_docker.sh` file to start the container, see this example

```bash
docker run \
    --name Mutils \
    --gpus '"device=0,1,2,3,4"' \
    --rm \
    -it \
    -v /path/to/the/slides:/home/input \
    -v /path/to/the/output:/home/output \
    -v /path/to/the/mutils/models:/home/models \
    --ulimit core=0 \
    szolgyen/mutils:v2 \
    bash
```
The container needs the

- /home/input
- /home/output
- /home/models

mounting points to be connected to the corresponding server volumes. Make sure that these are set properly in the `run_docker.sh` file.

#### Start the container

2. `./run_docker.sh`

Within the container, check and customize the configuration file at

3. `/home/MuTILs_Panoptic/configs/MuTILsWSIRunConfigs.yaml`.

> If not changing the parameters in the configuration file, MuTILs will run with the default parameters. The default parameters are found at [configs/MuTILsWSIRunConfigs.py](https://github.com/szolgyen/MuTILs_Panoptic/blob/520e1af15714abd9fae24cc9def5a07b5b6a6181/configs/MuTILsWSIRunConfigs.py#L145)

and run the MuTILsWSIRunner.py module to perform inference on your set of slides

4. `python MuTILs_Panoptic/mutils_panoptic/MuTILsWSIRunner.py`

### Recommended directory structure

```
Host (recommended)                      Container (default)
.                                          home
├── models                                  ├── models
│   ├── fold_1                              │   ├── fold_1
│   │    └── mutils_06022021_fold1.pt       │   │    └── mutils_06022021_fold1.pt
│   ├── fold_2                              │   ├── fold_2
│   │    └── mutils_06022021_fold2.pt       │   │    └── mutils_06022021_fold2.pt
│   ├── fold_3                              │   ├── fold_3
│   │    └── mutils_06022021_fold3.pt       │   │    └── mutils_06022021_fold3.pt
│   ├── fold_4                              │   ├── fold_4
│   │    └── mutils_06022021_fold4.pt       │   │    └── mutils_06022021_fold4.pt
│   └── fold_5                              │   └── fold_5
│        └── mutils_06022021_fold5.pt       │        └── mutils_06022021_fold5.pt
├── input                                   ├── input
├── output                                  ├── output
└── docker-compose.yaml                     ├── MuTILs_Panoptic
                                            └── venv
```

### Model weights

https://huggingface.co/mutils-panoptic/mutils/tree/main
