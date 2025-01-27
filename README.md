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

We recommend using the <i>szolgyen/mutils:v1</i> image from Docker Hub to perform inference with MuTILs. This image is based on Ubuntu 22.04 and includes a Python 3.10.12 virtual environment preconfigured with all the necessary packages for MuTILs. It is built on [nvidia/cuda:12.0.0-base-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.0.0-base-ubuntu22.04/images/sha256-3b6f49136ec6725b6fcc0fc04f2f7711d3d1e22d0328da2ca73e52dbd37fa4b1), providing CUDA 12.0.0 compatibility.

For a complete list of dependencies and additional details, refer to the [Dockerfile](https://github.com/szolgyen/MuTILs_Panoptic/blob/main/Dockerfile) in this repository.

#### Pull the docker image on your GPU server:

1. `docker pull szolgyen/mutils:v1`

We recommend using a `docker-compose.yaml` file to start the container, see this example

```yaml
version: '3'

services:
  mutilsdev:
    image: szolgyen/mutils:v1
    container_name: MutilsInference
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
    ipc: host
    network_mode: host
    volumes:
      - /your/path/to/the/model/weights:/home/models
      - /your/path/to/the/input/files:/home/input
      - /your/path/to/the/output/files:/home/output
    ulimits:
      core: 0
    stdin_open: true
    tty: true
    restart: "no"
```
The container needs the

- /home/models
- /home/input
- /home/output

mounting points to be connected to the corresponding server volumes. Make sure that these are set properly in the `docker-compose.yaml` file.

#### Start the container

2. `docker-compose up`

Once the container is up, attach to it in a separate terminal window:

3. `docker attach MutilsInference`

Within the container, check and customize the configuration file at

4. `/home/MuTILs_Panoptic/configs/MuTILsWSIRunConfigs.yaml`,

and run the MuTILsWSIRunner.py module to perform inference on your set of slides

5. `python MuTILs_Panoptic/mutils_panoptic/MuTILsWSIRunner.py`

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
