# ShuffleFAC: Ultra-Lightweight Ship-Radiated Sound Classification for Real-time Embedded Inference

[![Paper](https://img.shields.io/badge/Paper-IEEE_ESL_2026-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#)
[![Framework](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](#)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu%20%7C%20Docker%20%7C%20Raspberry%20Pi-green.svg)](#)

> **Official Implementation of "Ultra-Lightweight Ship-Radiated Sound Classification for Real-time Embedded Inference" (IEEE ESL 2026)**

ShuffleFAC is an ultra-lightweight acoustic model designed for **Underwater Acoustic Target Recognition (UATR)** on resource-constrained single-board computers (e.g., maritime monitoring buoys). By integrating **Frequency-Aware Convolution (FAC)** into an efficient channel-shuffling backbone, it maintains high classification accuracy while drastically reducing computational overhead.

---

## 📖 Research Background

With the rapid growth of maritime traffic, continuous monitoring of ship-radiated sound using autonomous buoy networks has become essential. However, deploying robust Deep Learning models on these edge platforms is highly challenging due to strict resource limits (battery power, lack of GPU, limited memory).
* **Limitation of Existing Models:** Standard lightweight architectures (like MobileNet and ShuffleNet) use shift-invariant convolutions, which are suboptimal for spectrograms where frequency position holds crucial semantic information. Additionally, complex network topologies (e.g., residual connections) incur high tensor-manipulation overhead on embedded CPUs.
* **Our Solution:** We propose **ShuffleFAC**, which explicitly encodes frequency-position information while utilizing a streamlined, linear backbone without residual paths to minimize both arithmetic (MACs) and non-arithmetic (memory allocation/copy) costs.

---

## 💡 Main Contributions

1. **Frequency-Adaptive Separable Convolution (FASC):** Injects learnable frequency-position encodings modulated by a lightweight self-attention branch prior to channel compression and depthwise convolution.
2. **Deployment-Aware Architecture:** Avoids residual and parallel paths to eliminate tensor-split/copy overhead, directly optimizing for on-device inference latency rather than just theoretical MACs.
3. **Extreme Efficiency on Edge:** **ShuffleFAC-8** achieves $69.69\%$ accuracy on the DeepShip dataset using only **11K parameters** and **1.06M MACs**, yielding a **0.58 ms** inference latency on a Raspberry Pi 5.

---

## 📊 Experimental Results (DeepShip Dataset)

Performance comparison on the DeepShip dataset (7:1:2 recording-level split, 128-band log-Mel spectrogram). Latency is measured on a Raspberry Pi 5 (Cortex-A76, fixed at 1.5 GHz).

| Model | Accuracy (%) | Parameters | MACs (M) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| MicroNet0 | 69.04 ± 0.46 | 379 K | 0.65 | 1.86 |
| ShuffleNet | 69.17 ± 3.37 | 919 K | 10.48 | 19.52 |
| **ShuffleFAC-8 (Ours)** | **69.69 ± 0.40** | **11 K** | **1.06** | **0.58** |
| **ShuffleFAC-16 (Ours)**| **69.68 ± 1.25** | **39 K** | **3.06** | **1.01** |
| **ShuffleFAC-32 (Ours)**| **72.45 ± 0.47** | **143 K** | **9.85** | **2.20** |

*ShuffleFAC-8 reduces inference latency by 68.8% and model size by 97.1% compared to MicroNet0 while maintaining comparable accuracy.*

---

## ⚙️ Requirements & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/KNU-LMAP/ShuffleFAC.git
cd ShuffleFAC
```

### 2. Environment Setup
You can set up the environment using `pip`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
---

## 📁 Dataset Preparation

We use the **DeepShip** dataset. For reliable generalization assessment, we apply a **recording-level split (7:1:2)** instead of segment-level splitting to prevent data leakage.

1. Download the DeepShip dataset.
2. Resample all audio files to `16 kHz`.
3. Generate 3-second non-overlapping clips and convert them to `128-band` log-Mel spectrograms.

---

## 🚀 Execution & Usage

### Training & Evaluation

```bash
python main.py
```
---

## 📎 Citation

If you find this code or our paper useful for your research, please consider citing our work:

```bibtex
@article{park2026ultra,
  title={Ultra-Lightweight Ship-Radiated Sound Classification for Real-time Embedded Inference},
  author={Park, Sangwon and Kim, Dongjun and Byun, Sung-Hoon and Park, Sangwook},
  journal={IEEE Embedded Systems Letters},
  year={2026},
  publisher={IEEE}
}
```

---

## 🤝 Acknowledgements

This research was supported by the Basic Science Research Program through the National Research Foundation of Korea (NRF) and other related research programs. 
