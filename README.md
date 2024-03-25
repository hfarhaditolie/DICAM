# DICAM: Deep Inception and Channel-wise Attention Modules for underwater image enhancement
[![Manuscript](https://www.hamidrezafarhadi.com/img/pdficon.png)](https://www.sciencedirect.com/science/article/pii/S0925231224003564)
[![GitHub stars](https://img.shields.io/github/stars/username/repository.svg?style=social)](https://github.com/hfarhaditolie/DICAM/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/username/repository.svg?style=social)](https://github.com/hfarhaditolie/DICAM/forks)

This repository contains the PyTorch implementation of the DICAM underwater image enhancement method provided by the authors.

---

> The light absorption and scattering of underwater impurities lead to poor underwater imaging quality. The existing data-driven based underwater image enhancement (UIE) techniques suffer from the lack of a large-scale dataset containing various underwater scenes and high-fidelity reference images. Besides, the inconsistent attenuation in different color channels and space areas is not fully considered for boosted enhancement. In this work, we constructed a large-scale underwater image (LSUI) dataset
> , and reported an U-shape Transformer network where the transformer model is for the first time introduced to the UIE task. The U-shape Transformer is integrated with a channel-wise multi-scale feature fusion transformer (CMSFFT) module and a spatial-wise global feature modeling transformer (SGFMT) module, which reinforce the network's attention to the color channels and space areas with more serious attenuation. Meanwhile, in order to further improve the contrast and saturation, a novel loss function combining RGB, LAB and LCH color spaces is designed following the human vision principle. The extensive experiments on available datasets validate the state-of-the-art performance of the reported technique with more than 2dB superiority.
> 
![Image Description](https://ars.els-cdn.com/content/image/1-s2.0-S0925231224003564-gr2_lrg.jpg)

![Image Description](https://ars.els-cdn.com/content/image/1-s2.0-S0925231224003564-gr3_lrg.jpg)



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install Awesome Project, follow these steps:
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

To use Awesome Project, execute the following command:

```bash
node awesome.js
