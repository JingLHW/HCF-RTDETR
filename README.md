# Enhanced Detection of Distracted Driving Behaviors via Cross-Scale Feature Fusion in RTDETR

This repository provides the official implementation of our proposed method **HCF-RTDETR**.

This repository corresponds to the following paper:

**Enhanced Detection of Distracted Driving Behaviors via Cross-Scale Feature Fusion in RTDETR**
*Submitted to The Visual Computer*

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

* a subset of the publicly available **State Farm Distracted Driver Detection (SFDDD)** dataset
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

## 📈 Experimental Results

### Performance Comparison on E-SFDDD

| Model          |     P (%) |     R (%) | mAP50 (%) | mAP50-95 (%) |
| -------------- | --------: | --------: | --------: | -----------: |
| Faster R-CNN   |     88.72 |     89.82 |     91.23 |        60.58 |
| YOLOv8         |     90.69 |     91.21 |     92.70 |        62.57 |
| YOLOv10        |     90.98 |     91.75 |     93.73 |        63.23 |
| YOLOv11        |     92.24 |     91.43 |     94.73 |        63.89 |
| PARE-YOLO      |     92.03 |     92.34 |     93.87 |        64.37 |
| DETR           |     90.22 |     90.91 |     92.23 |        62.12 |
| RTDETR         |     93.17 |     91.39 |     93.96 |        64.18 |
| RTDETR-r34     |     93.97 |     92.26 |     94.56 |        64.66 |
| RTDETR-r50     |     94.04 |     91.18 |     94.37 |        64.75 |
| **HCF-RTDETR** | **94.43** | **92.45** | **95.86** |    **65.43** |

### Ablation Study

| PConv | SFDConv | HFH-Net |     P (%) |    F1 (%) | mAP50 (%) | mAP50-95 (%) | GFLOPs | Params (M) |
| :---: | :-----: | :-----: | --------: | --------: | --------: | -----------: | -----: | ---------: |
|   ×   |    ×    |    ×    |     93.17 |     92.22 |     93.96 |        64.18 |   57.0 |      20.08 |
|   √   |    ×    |    ×    |     93.81 |     92.98 |     94.86 |        64.39 |   51.6 |      19.29 |
|   ×   |    √    |    ×    |     93.65 |     92.70 |     94.52 |        64.95 |   53.8 |      19.70 |
|   ×   |    ×    |    √    |     93.92 |     93.05 |     95.10 |        65.05 |   52.1 |      19.85 |
|   √   |    √    |    ×    |     94.19 |     92.86 |     95.08 |        65.62 |   50.0 |      18.87 |
|   √   |    √    |    √    | **94.43** | **93.43** | **95.86** |    **65.43** |   50.5 |      19.05 |

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
  title={Enhanced Detection of Distracted Driving Behaviors via Cross-Scale Feature Fusion in RTDETR},
  author={Yinglai Huang and Jing Wang and Wentao Gao and Xinyi Dong},
  year={2026},
  note={Submitted to The Visual Computer}
}

}
```

---

## 🔗 Code Availability

To ensure transparency and reproducibility, the source code of this work is publicly available at:

`https://github.com/JingLHW/HCF-RTDETR`

---

## 🙏 Acknowledgements

This work is built upon RTDETR and related open-source research. We sincerely thank the authors and contributors for their valuable work.

---

## 📧 Contact

For questions, collaborations, or dataset access requests, please contact:

**[nefuhyl@163.com](mailto:nefuhyl@163.com)**

Due to privacy considerations, access to certain data may be restricted.
