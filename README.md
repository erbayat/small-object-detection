# Small Object Detection Comparative Study

## Overview
Accurate detection of small objects (less than 20x20 pixels) remains a significant challenge in the field of deep learning, impacting critical applications such as UAV surveillance and traffic monitoring. This project aims to address these challenges by conducting a comparative study of two promising approaches:

1. Fine-tuning popular pre-trained models (e.g., YOLO, Faster R-CNN) specifically for small object datasets.
2. Evaluating the SAHI (Slicing Aided Hyper Inference) tiling approach to improve small object detection without model retraining.

## Objectives
- Optimize performance of pre-trained models through fine-tuning for small object datasets.
- Assess the effectiveness of the SAHI tiling approach in enhancing detection capabilities.
- Benchmark and compare the methods using standard metrics such as precision, recall, and mean Average Precision (mAP).

## Datasets and Resources
- **VisDrone Dataset**: A dataset comprising drone-captured images with small objects like pedestrians and vehicles, complete with bounding box annotations and object classes.
  - [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

## Methods
### 1. Fine-Tuning Pre-Trained Models
- **Models Used**: YOLO, Faster R-CNN
- **Approach**: Adaptation and fine-tuning of pre-trained object detection models using the VisDrone and SODA datasets to enhance their capability to detect small objects.

### 2. SAHI Tiling Approach
- **Methodology**: The SAHI framework slices large images into smaller, overlapping tiles, enabling better detection of small objects during inference.
- **Source**: [SAHI GitHub Repository](https://github.com/obss/sahi)

## Evaluation Metrics

- **Average Precision (AP)**
- **Average Recall (AR)**
- **Latency**

## Preliminary Results
| Model              | AP(.50-.95)-500 | AP(.50)-500 | AP(.75)-500 | AR(.50-.95)-1 | AR(.50-.95)-10 | AR(.50-.95)-100 | AR(.50-.95)-500 |
|:-------------------|----------------:|------------:|------------:|--------------:|---------------:|----------------:|----------------:|
| Yolo11n-B          |            4.27 |        8.16 |        3.98 |          1.64 |           5.65 |            6.41 |            6.41 |
| Yolo11n-FT         |           16.06 |       32.48 |       13.68 |          7.32 |          17.33 |           21.84 |           21.84 |
| Yolo11s-B          |            6.64 |       12.24 |        6.37 |          2.59 |           8.11 |            9.74 |            9.74 |
| Yolo11s-FT         |           18.39 |       37.81 |       15.24 |          8.32 |          20.29 |           25.43 |           25.43 |
| Yolo11m-B          |            7.31 |       13.15 |        7.17 |          2.85 |           8.83 |           10.43 |           10.43 |
| Yolo11m-FT         |           21.12 |       42.91 |       17.79 |          9.16 |          22.99 |           28.68 |           28.69 |
| Yolo11l-B          |            7.96 |       14.59 |        7.69 |          3.21 |           9.71 |           11.30 |           11.30 |
| Yolo11l-FT         |           20.67 |       42.33 |       17.14 |          9.25 |          22.86 |           28.80 |           28.80 |
| Yolo11x-B          |            7.45 |       13.47 |        7.27 |          3.05 |           9.24 |           10.73 |           10.73 |
| Yolo11x-FT         |           21.90 |       44.78 |       18.30 |          9.54 |          24.13 |           30.20 |           30.20 |


## Project Status
- The project is currently in the data preparation and initial implementation phase. Further updates will be provided as progress continues.

## References
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

## Contact
**Author**: Egemen Erbayat  
For any questions or discussions, please reach out through erbayat@gwu.edu 
