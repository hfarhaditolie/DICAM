<div align="center">

# DICAM: Deep Inception and Channel-wise Attention Modules for underwater image enhancement
[**Hamidreza Farhadi Tolie**](https://scholar.google.com/citations?user=nzCbjWIAAAAJ&hl=en&authuser=1)<sup>a</sup> · [**Jinchang Ren**](https://scholar.google.co.uk/citations?user=Vsx9P-gAAAAJ&hl=en)<sup>a</sup> · [**Eyad Elyan**](https://scholar.google.co.uk/citations?user=m3-aOvsAAAAJ&hl=en)<sup>b</sup>

<sup>a</sup> National Subsea Centre, Robert Gordon University, UK

<sup>b</sup> School of Computing, Robert Gordon University, UK

**Neurocomputing**

<hr>
<a href="https://www.sciencedirect.com/science/article/pii/S0925231224003564"><img src="https://www.hamidrezafarhadi.com/img/pdficon.png' alt='Paper PDF'></a>

</div>
This repository contains the PyTorch implementation of the DICAM underwater image enhancement method provided by the authors.

The manuscript is available at 
https://www.sciencedirect.com/science/article/pii/S0925231224003564 


:rocket:  :rocket:  :rocket: **News**:


- 2023/03/25 We have made our source codes with the pre-trained model on the UIEB dataset online 😊


## Abstract

> In underwater environments, imaging devices suffer from water turbidity, attenuation of lights, scattering, and particles, leading to low quality, poor contrast, and biased color images. This has led to great challenges for underwater condition monitoring and inspection using conventional vision techniques. In recent years, underwater image enhancement has attracted increasing attention due to its critical role in improving the performance of current computer vision tasks in underwater object detection and segmentation. As existing methods, built mainly from natural scenes, have performance limitations in improving the color richness and distributions we propose a novel deep learning-based approach namely Deep Inception and Channel-wise Attention Modules (DICAM) to enhance the quality, contrast, and color cast of the hazy underwater images. The proposed DICAM model enhances the quality of underwater images, considering both the proportional degradations and non-uniform color cast. Extensive experiments on two publicly available underwater image enhancement datasets have verified the superiority of our proposed model compared with several state-of-the-art conventional and deep learning-based methods in terms of full-reference and reference-free image quality assessment metrics.
---

![Image Description](https://ars.els-cdn.com/content/image/1-s2.0-S0925231224003564-gr2_lrg.jpg)

![Image Description](https://ars.els-cdn.com/content/image/1-s2.0-S0925231224003564-gr3_lrg.jpg)



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Feedback](#feedback)
- [License](#license)

## Installation

To install DICAM, follow these steps:
1. Clone the repository:

    ```bash
    git clone https://github.com/hfarhaditolie/DICAM
    ```

2. Navigate to the project directory:

    ```bash
    cd DICAM
    ```

3. Install dependencies:

    ```bash
    pip3 install -r requirements.txt
    ```

## Usage
To utilize the DICAM method for training, please follow these steps:

1. Download the UIEB and EUVP datasets from their respective sources.
   - For UIEB dataset, refer to the instructions provided in [UIEB dataset README](https://github.com/hfarhaditolie/DICAM/blob/main/Data/UIEB/readme.md).
   - For EUVP dataset, refer to the instructions provided in [EUVP dataset README](https://github.com/hfarhaditolie/DICAM/blob/main/Data/EUVP/readme.md).
   
2. Place the downloaded datasets in the 'Data/' directory of the DICAM repository, following the descriptions provided in each dataset's README.

3. Navigate to the 'UIEB/' directory for UIEB dataset or 'EUVP/' directory for EUVP dataset.

4. Run the _train_uieb.py_ script for UIEB dataset or _train_euvp.py_ script for EUVP dataset.

```bash
python3 UIEB/train_uieb.py
```
```bash
python3 EUVP/train_euvp.py
```

---
To utilize the DICAM method for testing, please follow these steps:

1. After downloading the datasets, navigate to the 'UIEB/' directory for UIEB dataset or 'EUVP/' directory for EUVP dataset.

2. Run the _test_uieb.py_ script for UIEB dataset or _test_euvp.py_ script for EUVP dataset.

```bash
python3 UIEB/test_uieb.py
```
```bash
python3 EUVP/test_euvp.py
```
---
To get the histogram evaluation, you need to run the _hist_distance.m_ script and specify the path for the generated enhanced images and their corresponding ground-truth ones.

## Citation
```bash
@article{TOLIE2024127585,
title = {DICAM: Deep Inception and Channel-wise Attention Modules for underwater image enhancement},
journal = {Neurocomputing},
pages = {127585},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.127585},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224003564},
author = {Hamidreza Farhadi Tolie and Jinchang Ren and Eyad Elyan},
keywords = {Underwater image enhancement, Deep learning, Inception module, Channel-wise attention module},
}
```
## Acknowledgement
We extend our gratitude to the creators of WaveNet for generously sharing their source code, which can be accessed [here](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration). This has greatly simplified the process of loading images from individual datasets.
## Feedback
If you have any enquires or feedback, please do not hesitate to contact us via @(h.farhadi-tolie@rgu.ac.uk, h.farhaditolie@gmail.com)
## License
This project is licensed under the [MIT License](LICENSE).
