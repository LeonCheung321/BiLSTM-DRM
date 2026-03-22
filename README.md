# BiLSTM-DRM: Bidirectional Residual Prediction Model for Short-Term GNSS-ZTD Gap Filling

A deep learning framework for high-accuracy imputation of short-term Zenith Total Delay (ZTD) data gaps in the Crustal Movement Observation Network of China (CMONOC), based on a Bidirectional Long Short-Term Memory (BiLSTM) residual prediction architecture with exponential backward error correction.

---

## Overview

This repository contains the complete implementation of the BiLSTM-DRM model presented in:

> **[Paper title to be confirmed upon acceptance]**  
> *[Journal name], [Year]*  
> DOI: [to be confirmed]

The model targets short-term ZTD data gaps of 2 to 6 hours — the most operationally frequent gap scenario in CMONOC — and achieves substantial accuracy improvements over established empirical models (HGPT2 and GPT3), with mean RMSE reductions exceeding 46% across all tested gap durations at held-out test stations.

---

## Repository Structure

```
project/
│
├── config.py                  # Centralised hyperparameter and path configuration
├── data_loader.py             # Data ingestion and parsing module
├── preprocessor.py            # Data preprocessing and feature engineering
├── model.py                   # BiLSTM-DRM model definition
├── trainer.py                 # Model training loop
├── tester.py                  # Evaluation and inference module
├── utils.py                   # Utility functions
├── diagnose.py                # Data diagnostics tool (recommended for first run)
├── check_imports.py           # Dependency and import verification tool
├── quick_test.py              # Lightweight end-to-end test (recommended before full run)
├── compare_results.py         # Cross-group performance comparison tool
├── visualize_correction.py    # Backward correction strategy visualisation
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Method Summary

BiLSTM-DRM models the residual between GNSS-observed ZTD and HGPT2-derived background ZTD estimates, rather than predicting raw ZTD directly. This residual decomposition isolates local tropospheric dynamics from the global background model bias, enabling more accurate short-term interpolation.

### Model Architecture

```
Input  (batch_size, seq_len=24, input_dim=11)
    │
    ▼
Bidirectional LSTM  (2 layers, hidden_size=128, dropout=0.2)
    │
    ▼
Fully Connected Layer 1  (256 → 128)  +  ReLU
    │
    ▼
Fully Connected Layer 2  (128 → 64)   +  ReLU
    │
    ▼
Fully Connected Layer 3  (64 → 1)
    │
    ▼
Output: predicted ZTD residual at next time step
    │
    ▼
Backward Exponential Correction
    │
    ▼
Reconstructed ZTD  =  Predicted Residual  +  HGPT2-ZTD
```

### Input Feature Vector (11-dimensional)

| Index | Feature | Description |
|-------|---------|-------------|
| 1 | `dZTD` | ZTD residual at current time step (observed ZTD − HGPT2-ZTD) |
| 2 | `ZTD_HGPT2` | HGPT2 background ZTD estimate (physical constraint) |
| 3 | `doy` | Day of year |
| 4 | `hod` | Hour of day |
| 5 | `lon` | Station longitude |
| 6 | `lat` | Station latitude |
| 7 | `height` | Station ellipsoidal height |
| 8 | `Z_nbr_mean` | Mean ZTD residual of 5 nearest neighbouring stations |
| 9 | `grad_lon` | ZTD residual gradient in the longitudinal direction |
| 10 | `grad_lat` | ZTD residual gradient in the latitudinal direction |
| 11 | `Z_nbr_range` | Range (max − min) of ZTD residuals among 5 neighbouring stations |

All features are standardised using a `StandardScaler` fitted on the training set.

### Backward Exponential Correction

After forward rolling prediction across the full gap, the cumulative error at the terminal boundary is redistributed to each predicted time step using an exponential weighting scheme:

```
ZTD_corr[i] = ZTD_pred[i] − ω_i × error_total

ω_i = (exp(λ · i/n_steps) − 1) / (exp(λ) − 1)
```

where `n_steps` is the total gap length and `λ` is a shaping parameter (default: 2.0) controlling the curvature of the weight distribution.

---

## Experimental Scope

This repository implements experiments for **short-term gap durations of 2 to 6 hours** at hourly resolution, corresponding to the results reported in the associated publication.

### Station Partition

| Group | Count | Role |
|-------|-------|------|
| Training-Validation Stations | 10 | Used for model training and internal validation |
| Held-out Test Stations | 10 | Withheld entirely from training; used for out-of-sample evaluation |

Stations are selected to span the principal hydroclimatic regimes of mainland China, including arid, semi-arid, humid subtropical, tropical monsoon, temperate semi-humid, and plateau environments.

### Evaluation Metrics

| Metric | Formula |
|--------|---------|
| RMSE | √(1/N · Σ(ŷᵢ − yᵢ)²) |
| MAE | 1/N · Σ\|ŷᵢ − yᵢ\| |

---

## Environment Requirements

```
Python  >= 3.8
PyTorch >= 1.10  (CUDA version recommended)
pandas
numpy
scikit-learn
openpyxl
```

### Installation

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install pandas numpy scikit-learn openpyxl

# Or install all dependencies at once
pip install -r requirements.txt
```

---

## Data Directory Structure

The expected data directory layout is as follows. Update `BASE_PATH` in `config.py` to match your local setup.

```
<BASE_PATH>/
│
├── stations.xlsx          # Station metadata (longitude, latitude, elevation)
├── RES_ZTD/               # Pre-computed ZTD residuals
│   ├── 2020/
│   └── 2021/
├── HGPT2/                 # HGPT2 model ZTD estimates
│   ├── 2020/
│   └── 2021/
├── GNSS/                  # GNSS-observed ZTD (ground truth)
│   ├── 2020/
│   └── 2021/
└── results/               # Model outputs (auto-created)
    ├── test_stations/     # Results for held-out test stations
    │   ├── gap_2h_gdzh_predictions.txt
    │   ├── gap_2h_metrics.txt
    │   └── ...
    └── train_val_stations/  # Results for training-validation stations
        ├── gap_2h_bjsh_predictions.txt
        ├── gap_2h_metrics.txt
        └── ...
```

### Input File Format

Station data files should be named as `<station_id><year>` (e.g., `bjsh2020`) with no file extension, space-delimited, in the following format:

```
year  doy  hod  minute  second  ZTD(mm)
2020  100  0    0       0       2450.23
2020  100  1    0       0       2451.12
...
```

---

## Usage

### Step 0: Verify dependencies

```bash
python check_imports.py
```

Resolve any reported import errors before proceeding.

### Step 1: Diagnose data (recommended for first run)

```bash
python diagnose.py
```

This tool checks:
- Directory structure and file presence
- File naming conventions and format
- Data readability and station metadata consistency
- Detection of anomalous ZTD values (e.g., ZTD = 0)

### Step 2: Quick end-to-end test (strongly recommended)

```bash
python quick_test.py
```

Runs a lightweight test using a subset of stations and 5 training epochs to verify that the full pipeline executes without errors. Estimated runtime: 5–10 minutes.

### Step 3: Configure hyperparameters and paths

Edit `config.py` to set your data paths and model hyperparameters:

```python
# Data path (modify to match your local directory)
BASE_PATH = r"C:\path\to\your\data"

# Model hyperparameters
HIDDEN_SIZE    = 128    # BiLSTM hidden state dimension
NUM_LAYERS     = 2      # Number of BiLSTM layers
DROPOUT        = 0.2    # Dropout rate
SEQ_LEN        = 24     # Input sequence length (hours)

# Training hyperparameters
BATCH_SIZE     = 64     # Mini-batch size
LEARNING_RATE  = 0.001  # Initial learning rate (Adam)
NUM_EPOCHS     = 100    # Maximum training epochs
WEIGHT_DECAY   = 1e-5   # L2 regularisation coefficient
PATIENCE       = 20     # Early stopping patience

# Backward correction
LAMBDA         = 2.0    # Exponential shaping parameter
```

### Step 4: Run the full experiment

```bash
python main.py
```

### Step 5: Compare results across station groups (optional)

```bash
python compare_results.py
```

Generates a console summary table and a detailed report file (`performance_summary.txt`) comparing training-validation and held-out test station performance.

---

## Output Files

### Prediction result file (`*_predictions.txt`)

```
year  doy  hod  GNSS_ZTD  HGPT2_ZTD  predicted_ZTD
2020  100  12   2450.23   2445.10    2448.56
2020  100  13   2451.12   2446.20    2449.87
...
```

### Residual result file (`*_residuals.txt`)

```
year  doy  hod  true_residual  predicted_residual
2020  100  12   5.13           3.46
2020  100  13   4.92           3.81
...
```

### Evaluation metrics file (`*_metrics.txt`)

```
Experiment: gap_2h
============================================================
Mean RMSE  :  15.2345 mm
Mean MAE   :  12.3456 mm
N samples  :  1200
```

### Training output files

| File | Content |
|------|---------|
| `experiment_log.txt` | Full experiment log with timestamps |
| `training_losses.txt` | Per-epoch training and validation loss |
| `bilstm_drm.pth` | Saved model weights |

---

## Key Hyperparameter Reference

| Parameter | Value | Basis |
|-----------|-------|-------|
| `seq_len` | 24 | Aligned with ZTD diurnal cycle |
| `hidden_size` | 128 | Cross-validation; computational efficiency |
| `num_layers` | 2 | Capacity vs. overfitting trade-off |
| `dropout` | 0.2 | Validation loss convergence |
| `batch_size` | 64 | GPU memory and gradient stability |
| `learning_rate` | 0.001 | Adam default; validated on held-out set |
| `weight_decay` | 1e-5 | L2 regularisation |
| `patience` | 20 | Early stopping on validation plateau |
| `λ` | 2.0 | Empirically determined |

---

## Troubleshooting

### ImportError on module load

```bash
python check_imports.py
```

Ensure all `.py` files are present in the project directory and contain no syntax errors.

### Zero stations loaded (`"Successfully loaded 0 stations"`)

1. Run `python diagnose.py` to identify the issue.
2. Verify file naming: files must follow the format `<station_id><year>` with **no extension** (e.g., `bjsh2020`, not `bjsh2020.txt`).
3. Verify file content format: space-delimited, no BOM, UTF-8 or ASCII encoding.
4. Verify `BASE_PATH` in `config.py` using double backslashes or a raw string: `r"C:\path\to\data"`.

### CUDA not available

```python
import torch
print(torch.cuda.is_available())   # Should return True
print(torch.version.cuda)          # Should match your driver version
```

Reinstall PyTorch with the correct CUDA version from https://pytorch.org/get-started/locally/.

### Out-of-memory error

Reduce batch size or hidden state dimension in `config.py`:

```python
BATCH_SIZE  = 32   # Reduce from 64
HIDDEN_SIZE = 64   # Reduce from 128
```

### Slow convergence

```python
LEARNING_RATE = 0.0001   # Reduce learning rate
WEIGHT_DECAY  = 1e-4     # Increase regularisation
```

---

## Data Availability

CMONOC GNSS tropospheric delay data are accessible through the GNSS Data Product Service Platform of the China Earthquake Administration at [ftp.cgps.ac.cn](ftp://ftp.cgps.ac.cn).

---

## Citation

If you use this code in your research, please cite the associated publication:

```bibtex
@article{[citation key to be confirmed],
  title   = {[Paper title to be confirmed]},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

---

## License

This code is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) licence. You are free to use, share, and adapt this material for any purpose, provided appropriate credit is given to the original authors.

---

## Acknowledgements

The authors gratefully acknowledge the GNSS Data Product Service Platform of the China Earthquake Administration (ftp.cgps.ac.cn) for providing the CMONOC tropospheric delay data. The HGPT2 model (Mateus et al., 2021) and GPT3 model (Landskron & Böhm, 2018) were used as empirical baselines and are publicly available from their respective developers.
