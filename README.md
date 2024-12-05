# Small Object Detection Comparative Study

## Contents

- [Small Object Detection Comparative Study](#small-object-detection-comparative-study)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Objectives](#objectives)
  - [Datasets](#datasets)
  - [Methods](#methods)
    - [1. Fine-Tuning Pre-Trained Models](#1-fine-tuning-pre-trained-models)
    - [2. SAHI Tiling Approach](#2-sahi-tiling-approach)
  - [Evaluation and Metrics](#evaluation-and-metrics)
    - [Warmup and Experimentation](#warmup-and-experimentation)
    - [Metrics](#metrics)
  - [Installation](#installation)
  - [Run Experiments](#run-experiments)
  - [Latency Results](#latency-results)
  - [Accuracy Results](#accuracy-results)
  - [References](#references)
  - [Contact](#contact)
    
## Overview
Accurate detection of small objects (less than 20x20 pixels) remains a significant challenge in the field of deep learning, impacting critical applications such as UAV surveillance and traffic monitoring. This project aims to address these challenges by conducting a comparative study of two promising approaches:

1. Fine-tuning popular pre-trained models (e.g., YOLO) specifically for small object datasets.
2. Evaluating the SAHI (Slicing Aided Hyper Inference) tiling approach to improve small object detection without model retraining.

## Objectives
- Optimize performance of pre-trained models through fine-tuning for small object datasets.
- Assess the effectiveness of the SAHI tiling approach in enhancing detection capabilities.
- Benchmark and compare the methods using standard metrics such as Average Precision (AP) and Average Recall (AR).

## Datasets
- **VisDrone Dataset**: A dataset comprising drone-captured images with small objects like pedestrians and vehicles, complete with bounding box annotations and object classes.
  - [VisDrone2019-VID-test-dev](https://github.com/VisDrone/VisDrone-Dataset) (17 clips, 6635 frames)

## Methods
### 1. Fine-Tuning Pre-Trained Models
- **Models Used**: YOLO11 
- **Approach**: Adaptation and fine-tuning of pre-trained object detection models using the VisDrone dataset to enhance their capability to detect small objects.

### 2. SAHI Tiling Approach
- **Methodology**: The SAHI framework slices large images into smaller, overlapping tiles, enabling better detection of small objects during inference.
- **Source**: [SAHI GitHub Repository](https://github.com/obss/sahi)
- **Improvement**: Yolo11 support is added to the framework.

## Evaluation and Metrics

### Warmup and Experimentation
- **Warmup Runs:** Before conducting experiments, warmup runs are performed to fully utilize GPU performance and ensure consistent results.
- **Experimentation:** Each experiment is repeated multiple times, and latency is recorded for every run. This approach provides a more reliable understanding of performance.

### Metrics
1. **Average Precision (AP):** Evaluated using the [Visdrone Toolkit](https://github.com/VisDrone/VisDrone2018-VID-toolkit).
2. **Average Recall (AR):** Calculated with the [Visdrone Toolkit](https://github.com/VisDrone/VisDrone2018-VID-toolkit).
3. **Latency:** Measured to assess the computational efficiency of the approach.

   
## Installation

1. Create a `conda environment`:

```bash
conda create -n small_object_detection python=3.11 -y
conda activate small_object_detection
```

2. Install the environment:

```bash
git clone https://github.com/erbayat/small-object-detection.git
cd small_object_detection
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt 
```

3. Download Models and Dataset:

```bash
# This script downloads the necessary models and datasets

# Download models and datasets using the combined script
bash ./scripts/download_files.sh  # Executes the Bash script to download all files

# OR

# Download models separately
python3 ./src/download_models.py  # This script handles downloading the models

# Download datasets separately
python3 ./src/download_dataset.py  # This script handles downloading the datasets
```

## Run Experiments

```bash
# Option 1: Run evaluation using the pre-written Bash script
# This script automates the evaluation process for all models and configurations
bash ./scripts/run_eval.sh

# OR

# Option 2: Run evaluation for a specific model using the Python script

# Run evaluation for the specified model without SAHI 
python3 ./src/eval_models.py --model_name selected_model_name_here

# Run evaluation for the specified model with SAHI 
python3 ./src/eval_models.py --model_name selected_model_name_here --is_sahi True
```


## Latency Results
| Method             |   Average Latency (ms) |   Standard Deviation (ms) |
|:----------------------|-----------------------:|--------------------------:|
| Yolo11n-B               |                   28   |                      10.5 |
| Yolo11n-FT              |                   27.9 |                      10.4 |
| Yolo11n-B++             |                  243.2 |                     105.4 |
| Yolo11n-FT++            |                  258.5 |                     106.8 |
| Yolo11s-B               |                   28.1 |                      10.5 |
| Yolo11s-FT              |                   28   |                      10.5 |
| Yolo11s-B++             |                  246.8 |                     104.3 |
| Yolo11s-FT++            |                  262.8 |                     106.1 |
| Yolo11m-B               |                   32.7 |                      10.7 |
| Yolo11m-FT              |                   32.5 |                      10.6 |
| Yolo11m-B++             |                  299.2 |                     127.9 |
| Yolo11m-FT++            |                  315.2 |                     128.8 |
| Yolo11l-B               |                   34.6 |                      10.9 |
| Yolo11l-FT              |                   34.7 |                      10.7 |
| Yolo11l-B++             |                  327.9 |                     142.9 |
| Yolo11l-FT++            |                  344.6 |                     143.1 |
| Yolo11x-B               |                   44.3 |                      10.6 |
| Yolo11x-FT              |                   44.2 |                      10.7 |
| Yolo11x-B++             |                  466.1 |                     208.5 |
| Yolo11x-FT++            |                  481.3 |                     208.4 |

- **-B**: Pretrained weights.
- **-FT**: Fine-tuned with the Visdrone dataset.
- **++**: SAHI (Sliced Aided Hyper Inference) approach is applied.

> **Note:** All latency results were obtained using an RTX 2070 Ti GPU.
> 
## Accuracy Results
| Method             | AP(.50-.95)-500 | AP(.50)-500 | AP(.75)-500 | AR(.50-.95)-1 | AR(.50-.95)-10 | AR(.50-.95)-100 | AR(.50-.95)-500 |
|:----------------------|----------------:|------------:|------------:|--------------:|---------------:|----------------:|----------------:|
| Yolo11n-B               |            4.27 |        8.16 |        3.98 |          1.64 |           5.65 |            6.41 |            6.41 |
| Yolo11n-FT              |           16.06 |       32.48 |       13.68 |          7.32 |          17.33 |           21.84 |           21.84 |
| Yolo11n-B++             |            8.49 |       17.37 |        7.38 |          3.08 |           9.73 |           13.38 |           13.38 |
| Yolo11n-FT++            |           16.94 |       34.99 |       14.22 |          8.22 |          20.77 |           25.97 |           25.97 |
| Yolo11s-B               |            6.64 |       12.24 |        6.37 |          2.59 |           8.11 |            9.74 |            9.74 |
| Yolo11s-FT              |           18.39 |       37.81 |       15.24 |          8.32 |          20.29 |           25.43 |           25.43 |
| Yolo11s-B++             |            9.64 |       19.05 |        8.68 |          3.73 |          11.09 |           14.59 |           14.59 |
| Yolo11s-FT++            |           19.55 |       40.79 |       16.18 |          8.79 |          23.35 |           29.57 |           29.57 |
| Yolo11m-B               |            7.31 |       13.15 |        7.17 |          2.85 |           8.83 |           10.43 |           10.43 |
| Yolo11m-FT              |           21.12 |       42.91 |       17.79 |          9.16 |          22.99 |           28.68 |           28.69 |
| Yolo11m-B++             |           10.80 |       21.28 |        9.53 |          4.34 |          12.25 |           15.79 |           15.79 |
| Yolo11m-FT++            |           21.88 |       45.49 |       18.20 |          9.77 |          25.54 |           32.14 |           32.17 |
| Yolo11l-B               |            7.96 |       14.59 |        7.69 |          3.21 |           9.71 |           11.30 |           11.30 |
| Yolo11l-FT              |           20.67 |       42.33 |       17.14 |          9.25 |          22.86 |           28.80 |           28.80 |
| Yolo11l-B++             |           11.19 |       21.88 |        9.98 |          4.60 |          12.54 |           16.15 |           16.15 |
| Yolo11l-FT++            |           21.60 |       44.71 |       17.96 |          9.83 |          25.17 |           31.88 |           31.91 |
| Yolo11x-B               |            7.45 |       13.47 |        7.27 |          3.05 |           9.24 |           10.73 |           10.73 |
| Yolo11x-FT              |           21.90 |       44.78 |       18.30 |          9.54 |          24.13 |           30.20 |           30.20 |
| Yolo11x-B++             |           10.78 |       21.06 |        9.57 |          4.33 |          12.57 |           16.16 |           16.16 |
| Yolo11x-FT++            |           21.89 |       45.71 |       18.31 |          9.83 |          25.83 |           32.67 |           32.70 |

- **-B**: Pretrained weights.
- **-FT**: Fine-tuned with the Visdrone dataset.
- **++**: SAHI (Sliced Aided Hyper Inference) approach is applied.

## References
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Yolo11](https://docs.ultralytics.com/models/yolo11/)

## Contact
**Author**: Egemen Erbayat  
For any questions or discussions, please reach out through erbayat@gwu.edu 
