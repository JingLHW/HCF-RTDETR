# HCF-RTDETR: High-Frequency Guided Cross-Scale Fusion for Distracted Driving Detection

This repository provides the official implementation of our proposed method **HCF-RTDETR**.

This repository corresponds to the following paper:

**HCF-RTDETR: High-Frequency Guided Cross-Scale Fusion for Distracted Driving Detection**



---

## 🚀 Overview

Distracted driving is a major contributor to traffic accidents and poses significant risks to road safety. Detecting such behaviors in in-cabin environments is challenging due to small targets, occlusion, pose variation, and illumination changes.

To address these challenges, we propose **HCF-RTDETR**, an enhanced detection framework based on **RTDETR**. The method introduces high-frequency guided cross-scale feature fusion to improve feature representation and robustness while maintaining efficient computation.

---

## 🧠 Method

The proposed **HCF-RTDETR** is built upon **RTDETR-r18** and consists of two key components:

### 1. P-SFD Backbone

* **PConv_Block**

  * Enhances shallow texture and edge features
  * Reduces redundant computation
  * Improves fine-grained feature representation

* **SFDConv**

  * Combines spatial-domain and frequency-domain modeling
  * Improves robustness under occlusion, pose variation, and illumination changes
  * Enhances representation of driver-related regions

### 2. HFH-Net

* **HiLo Attention**

  * Captures high-frequency local details
  * Models low-frequency global dependencies

* **SlimNeck-ASF**

  * Enables adaptive multi-scale feature fusion
  * Improves feature propagation across scales
  * Reduces missed detections of weak and small targets

---

## 📁 Project Structure

```text
HCF-RTDETR/
├── dataset/                 # dataset structure only (no images included)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── configs/
│   ├── data.yaml
│   └── HCF.yaml
├── models/
│   ├── attention.py
│   ├── block.py
│   ├── head.py
│   └── transformer.py
├── train.py
├── val.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/JingLHW/HCF-RTDETR.git
cd HCF-RTDETR

conda create -n hcf-rtdetr python=3.10 -y
conda activate hcf-rtdetr

pip install -r requirements.txt
```

---

## 📊 Dataset

We conduct experiments on **E-SFDDD**, which is constructed from:

* a subset of the publicly available **State Farm Distracted Driver Detection (SFDDD)** dataset ([official source](https://www.kaggle.com/c/state-farm-distracted-driver-detection))
* additional in-cabin samples collected by the authors

The dataset contains five classes:

* Normal Driving
* Phone Use
* Calling
* Drinking
* Operating Device

### Dataset Structure

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

> Note: Only the dataset structure is provided in this repository. No images or annotations are included.

### Availability

* The original **SFDDD** dataset is publicly available from its official source.
* The self-collected in-cabin data is **not publicly released** due to:

  * privacy considerations
  * data protection regulations
  * redistribution restrictions

Therefore, the complete E-SFDDD dataset used in this work is **not publicly available**.

---

## 🏋️ Reproducibility

To ensure transparency and reproducibility, this repository provides:

* the full implementation of HCF-RTDETR
* training and validation scripts
* configuration files
* dataset structure and annotation format

### Quick Start

```bash
pip install -r requirements.txt
python train.py
python val.py
```

---

## 🏋️ Training

Before training, please ensure that the dataset path in `configs/data.yaml` is correctly configured.

```bash
python train.py
```

or

```bash
python train.py --data configs/data.yaml --cfg configs/HCF.yaml
```

---

## 🔍 Validation

```bash
python val.py
```

or

```bash
python val.py --data configs/data.yaml --cfg configs/HCF.yaml
```

---



## 📦 Weights

Pretrained weights are **not included** in this repository.

They will be released after the paper is accepted.

---

## 📌 Notes

* Please avoid using absolute dataset paths.
* Large files such as datasets, weights, and logs are not included.
* Please ensure that the dataset organization matches the settings in `configs/data.yaml`.

---

## 📖 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{huang2026hcfrtdetr,
  title={HCF-RTDETR: High-Frequency Guided Cross-Scale Fusion for Distracted Driving Detection},
  author={Yinglai Huang and Jing Wang and Wentao Gao and Xinyi Dong},
  year={2026},
  note={Submitted to Image and Vision Computing}
}



```

---

## 🔗 Code Availability

To ensure transparency and reproducibility, the source code of this work is publicly available at:

`https://github.com/JingLHW/HCF-RTDETR`

---

## 🙏 Acknowledgements

This work was supported by the National Natural Science Foundation of China (Grant No. 32271781). We are also grateful to the contributors of the RTDETR framework and other open-source projects that made this research possible.

---

## 📧 Contact

For questions, collaborations, or dataset access requests, please contact:

**[nefuhyl@163.com](mailto:nefuhyl@163.com)**

Due to privacy considerations, access to certain data may be restricted.
