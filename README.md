# Assignment 2 Deep Learning for Perception

**Name:** [ADD YOUR NAME]  
**Roll number:** 23F-0683  
**Section:** [ADD YOUR SECTION]  
**GitHub repository:** [PASTE YOUR REPOSITORY URL]

## Overview

This assignment investigates CIFAR-10 classification with limited training data
using a custom Inception-style CNN. It compares six augmentation configurations,
examines errors under image corruption, and evaluates test-time augmentation.

## Repository files

| File | Description |
|---|---|
| [23F-0683_DL_ASS02_7B.ipynb](23F-0683_DL_ASS02_7B.ipynb) | Executed Section B notebook containing code, results and plots |
| [23F-0683_DL_ASS02_7B.docx](23F-0683_DL_ASS02_7B.docx) | One-page Word report containing the final results and repository link |
| README.md | Project overview and reproduction instructions |

## Dataset and reproducibility

- Dataset: CIFAR-10 loaded through `torchvision`.
- Select 500 original training images, exactly 50 per class.
- Split into 400 training images and 100 validation images, with 40/10 per class.
- Use the original 10,000 test images for clean and local shifted evaluation.
- Subset and split seeds: **0**. Training seeds: **0, 1, 2**.
- Local shifted corruption seed: **1234**. Grayscale-only seed: **7**.
- Error-sampling seed: **0**. Corruption processing batch size: **1000**.
- Normalization uses the 400 training images only. Training, early stopping and
  hyperparameter selection use training/validation data, not test scores.
- Test scores are displayed before later tuning in this notebook, although the
  visible selection code uses validation accuracy.

## Model and training

The network has **119,210 trainable parameters**, two Inception modules,
BatchNorm, global average pooling, Dropout before the final classifier, and
randomly initialized weights. Training uses AdamW, batch size 64, OneCycleLR
with maximum learning rate **0.003**, and early stopping with patience **20**.
Each run is capped at **60 epochs**.

A validation search selects **Dropout 0.1** and **weight decay 0.005** for the
six ablations. The final configuration is **No augmentation**, selected by the
highest mean validation accuracy among these configurations.

## Results

Accuracy is reported as mean ± sample standard deviation over three training
seeds (`ddof=1`). The initial baseline and final ablation model use different
regularization settings and should be compared separately.

| Evaluation | Initial baseline | Final model |
|---|---:|---:|
| Validation | 51.00 ± 4.58% | 47.00 ± 4.36% |
| Clean test | 47.02 ± 1.28% | 44.92 ± 3.01% |
| Local shifted test | 27.61 ± 0.29% | 25.87 ± 2.72% |

Mixup plus label smoothing has the highest observed local shifted mean,
**32.85 ± 2.50%**, but it is not the validation-selected final model. Six-view
TTA gives approximately **26.23 ± 2.86%** for the final model, a mean gain of
**0.36 percentage points** at approximately **6.1×** recorded inference time.
The TTA aggregate is calculated from the notebook's rounded per-seed outputs.

## Reproduce the experiment

1. Open `23F-0683_DL_ASS02_7B.ipynb` in Google Colab and select a GPU runtime.
2. Use the working Colab installation of PyTorch and torchvision. The notebook
   also requires NumPy, pandas and Matplotlib. If an import is missing, install
   the missing package in a notebook cell before running the experiment.
3. Place `cifar-10-python.tar.gz` at the top level of your Google Drive, or edit
   the Drive source path in the notebook to match its location. The archive's
   expected MD5 is `c58f30108f718f92721af3b95e74349a`.
4. Run the nonempty notebook cells in order. Its preparation cells mount Drive,
   copy the archive into `./data`, and verify the checksum. Alternatively, skip
   those three preparation cells if torchvision's automatic download works.
5. Keep `USE_PROVIDED = False` to use torchvision CIFAR-10 and the documented split.
6. Run the baseline, validation search, six ablations, error analysis,
   grayscale-only evaluation and TTA. This performs 27 training runs in total:
   3 initial baseline, 6 tuning and 18 ablation runs.
7. Save the executed notebook with its results and plots visible.

The recorded execution reports **Python 3.13.15**, **PyTorch 2.11.0+cu130**, and
**CUDA**; notebook metadata indicates a T4 runtime. Other package versions were
not printed. Different software versions or CPU/GPU random draws can change results.

## Corruption scope

The local shifted pipeline uses rotation up to **±25°**, brightness and contrast
factors **0.5–1.5**, a constant-0.5 occlusion mask, and random grayscale with
probability **0.3**. The nominal 25% mask is a 17×17 square clipped at image
boundaries, so its actual coverage varies. Blur and noise are training augmentations.

The instructor's exact additional hidden corruption was not supplied. Grayscale
is an explicitly documented local assumption; these scores are **not official
hidden-test results**. The notebook includes all recorded figures and outputs,
so separate plot, dataset or checkpoint files are not required in this repository.
