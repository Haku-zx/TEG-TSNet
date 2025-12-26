# TEG-TSNet

**TEG-TSNet: Tensor-Evolving Graph with Temporal Separation Network for Spatiotemporal Forecasting**

This repository provides the official PyTorch implementation of **TEG-TSNet**, a spatiotemporal forecasting model proposed in our paper.
The model is designed for traffic flow prediction and other spatiotemporal sequence forecasting tasks under **dynamic graph structures**.

---

## 📌 Overview

Spatiotemporal forecasting over traffic networks requires modeling:

* Dynamic and time-varying spatial dependencies
* Temporal evolution of node interactions
* Trend–seasonal disentanglement in time series
* Heterogeneous temporal patterns across nodes

To address these challenges, **TEG-TSNet** introduces:

* **Tensor-Based Evolving Graph Generation (TB2G)**
* **Diffusion Graph Encoder (DGE)** with dynamic adjacency
* **Trend–Seasonal Decomposition (TSD)** with dual GRU encoders
* **Spatiotemporal Attention Fusion (SAF)**
* **Sparse Mixture-of-Experts Decoder (SMoE)**

The proposed framework jointly captures **temporal dynamics**, **evolving spatial structures**, and **heterogeneous prediction patterns**.

---

## 🧠 Model Architecture

The overall architecture consists of five main components:

1. **Graph Spectral Embedding (GSE)**
   Encodes spatial priors using Laplacian positional encodings.

2. **Temporal Embedding & Trend–Seasonal Decomposition (TSD)**
   Decomposes input signals into trend and seasonal components guided by temporal embeddings.

3. **Tensor-Based Evolving Graph Generator (TB2G)**
   Dynamically constructs time-dependent adjacency matrices via tensor factorization.

4. **Diffusion Graph Encoder (DGE)**
   Performs multi-order diffusion convolution over dynamic graphs.

5. **SMoE Decoder with Spatiotemporal Attention Fusion (SAF)**
   Captures heterogeneous spatiotemporal dependencies and improves multi-step prediction.

---

## 📁 Repository Structure

```
TEG-TSNet/
│
├── model.py            # Core implementation of TEG-TSNet
├── train.py            # Training and evaluation pipeline
├── utils.py            # Utility functions (metrics, graph ops, LapPE)
├── graph_utils.py      # Graph construction utilities
├── metrics.py          # MAE / RMSE / MAPE
├── prepareData.py      # Data preprocessing
│
├── conf/               # Configuration files
│   ├── PEMSD4_1dim_12.conf
│   ├── PEMSD8_1dim_12.conf
│   └── JiNan_1dim_12.conf
│
├── data/               # Small datasets used in experiments
│   ├── PEMS04/
│   │   └── PEMS04.csv
│   ├── PEMS08/
│   │   └── PEMS08.csv
│   └── JiNan/
│       └── JiNan.csv
│
└── README.md
```

---

## 🛠️ Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.10
* NumPy
* SciPy
* scikit-learn
* torch-geometric

You can install the required packages via:

```bash
pip install numpy scipy scikit-learn torch torch-geometric
```

---

## 🚀 Running the Code

### 1️⃣ Data Preparation (optional)

If you want to regenerate training/validation/test splits:

```bash
python prepareData.py
```

By default, processed data will be stored in `.npz` format.

---

### 2️⃣ Training

Run training with a specified configuration file:

```bash
python train.py --config conf/PEMSD4_1dim_12.conf
```

Other available configs:

```bash
conf/PEMSD8_1dim_12.conf
conf/JiNan_1dim_12.conf
```

---

### 3️⃣ Evaluation

After training, the script automatically reports:

* MAE
* RMSE
* MAPE
* Per-horizon forecasting performance

Intermediate representations (e.g., dynamic adjacency matrices, hidden states) are saved for analysis.

---

## 📊 Datasets

We provide **small-sized versions** of the following datasets for reproducibility:

* **PEMS04**
* **PEMS08**
* **JiNan traffic dataset**

These datasets are included **only for experimental reproducibility** and academic use.

---

## 🔍 Reproducibility Notes

* All paths are **relative paths** (no absolute paths required).
* Random seeds are fixed where applicable.
* Dynamic adjacency matrices can be extracted from saved intermediate outputs.
* The implementation follows the model description in the paper.

---

## 📄 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{
  title   = {TEG-TSNet: Tensor-Evolving Graph with Temporal Separation Network for Spatiotemporal Forecasting},
  author  = {Anonymous},
  journal = {Under Review},
  year    = {2025}
}
```

---

## 📬 Contact

For questions or issues, please open an issue in this repository.

---


