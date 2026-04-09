# Pretrained Model for 3D T1-weighted Brain MRI Quality Control Trained on a Large Dataset

This project provides a **pretrained model for automatic quality control (MRI QC) of 3D T1-weighted brain MRI**, trained on a **large dataset**, and supports **few-shot fine-tuning based on synthetic artifact generation**.

The method combines:

- **Pretraining on a large dataset**
- **Synthetic artifact augmentation**
- **Few-shot fine-tuning**
- **Model ensemble prediction**
---

## Environment Setup

It is recommended to use **Python 3.11**.

Create the environment:

```bash
conda create -n brain_t1_qc python=3.11 -y
conda activate brain_t1_qc
```

Install preprocessing and model dependencies:

```bash
pip install -r requirements_preprocess.txt
pip install -r requirements_model.txt
```

### GPU Version (Recommended: CUDA 12.4)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### CPU Version

```bash
pip install torch torchvision torchaudio
```

---

## Required Neuroimaging Software

The preprocessing pipeline requires the following neuroimaging tools:

- **FSL** (Recommended version: 6.0.4)  
- **FreeSurfer** (Recommended version: 7.4.1)

Please ensure they are properly installed and available in the system **PATH**.

---

## Overall Workflow

```text
Data Preprocessing
        ↓
Data Augmentation
        ↓
Model Pretraining
        ↓
Transfer Learning (Fine-tuning)
        ↓
Model Prediction
```

---

## Data Preprocessing

### Data Directory Structure

Original raw data:

```
data/images/raw/original
```

Augmented raw data:

```
data/images/raw/augmented
```

Preprocessed data:

```
data/images/preprocessed
```

The final file used by the model is:

```
T1_resize.nii.gz
```

After preprocessing, **synthetic artifacts can be added during either pretraining or fine-tuning**.

Pretrained model checkpoints are provided in:

```
checkpoints/pretrained_model
```

---

### Split Files

Prepared CSV split files are located in:

```
data/splits
```

These CSV files should be converted into **PKL format** using:

```
tools/conversion
```

---

### Training and Prediction Scripts

Pretraining scripts:

```
pretrain_train.py
pretrain_predict.py
```

Fine-tuning scripts:

```
transfer_train.py
transfer_predict.py
```

---

### Fine-tuning Configuration Templates

Template files:

```
checkpoints/transfer_yaml_template
```

Batch YAML configuration files can be generated using:

```
tools/transfer_templates
```

---

### Ensemble Prediction

After training, ensemble prediction results can be summarized using:

```
tools/ensemble/ensemble_summary.py
```

---

### Preprocessing Pipelines

Preprocessing pipeline for **original data**:

```
preprocess/preprocessing
```

Preprocessing pipeline for **augmented data**:

```
preprocess/augmentation/processing
```

> **Note**
> The preprocessing pipeline requires **FSL** and **FreeSurfer**.

---

## Data Augmentation

When the number of **high-quality MRI scans is limited**, data augmentation can be used to generate additional samples.

These augmented samples can **replace or supplement real samples during the fine-tuning stage**.

Augmentation script:

```
preprocess/augmentation/generate/augment.py
```

---

## Model Pretraining

Training script:

```
pretrain_train.py
```

Prediction script:

```
pretrain_predict.py
```

Configuration files:

```
configs/pretrain
```

Pretrained model checkpoint:

```
checkpoints/pretrained_model/chort_a_best
```

---

## Transfer Learning (Fine-tuning)

Training script:

```
transfer_train.py
```

Prediction script:

```
transfer_predict.py
```

Configuration files:

```
configs/transfer
```

Batch generation of configuration files:

```
tools/transfer_templates
```

---

## Utility Tools

This project provides a set of utility tools for **data preparation, configuration generation, and result aggregation**, all located in the `tools/` directory.

---

### CSV to PKL Conversion

Before model training, CSV files need to be converted into **PKL format**.

Location:

```
tools/conversion
```

---

### Batch Generation of Fine-tuning Configurations

To facilitate large-scale transfer learning experiments, configuration files can be automatically generated using:

```
tools/transfer_templates
```

---

### Ensemble of Fine-tuning Results

After fine-tuning, multi-fold prediction results can be aggregated using:

```
tools/ensemble/ensemble_summary.py
```

---

### Batch Execution

Batch training and prediction can be executed with:

```bash
nohup python tools/batch/batch_transfer_multi_model.py > logs/nohup_transfer.log 2>&1 &
nohup python tools/batch/batch_predict_multi_model.py > logs/nohup_predict.log 2>&1 &
```
