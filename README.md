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
- **SODA-D and SODA-A Datasets**: Benchmark datasets specifically designed for small object detection, useful for comparing the performance of various detection models.
  - [IEEE SODA Datasets Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10168277)

## Methods
### 1. Fine-Tuning Pre-Trained Models
- **Models Used**: YOLO, Faster R-CNN
- **Approach**: Adaptation and fine-tuning of pre-trained object detection models using the VisDrone and SODA datasets to enhance their capability to detect small objects.

### 2. SAHI Tiling Approach
- **Methodology**: The SAHI framework slices large images into smaller, overlapping tiles, enabling better detection of small objects during inference.
- **Source**: [SAHI GitHub Repository](https://github.com/obss/sahi)

## Evaluation Metrics
- **Precision**
- **Recall**
- **Mean Average Precision (mAP)**
- **Latency**

## Results
This section will be updated with detailed benchmark results, comparative charts, and key findings upon completion of the experiments.

## Project Status
- The project is currently in the data preparation and initial implementation phase. Further updates will be provided as progress continues.

## References
- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [SODA Datasets - IEEE Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10168277)

## Contact
**Author**: Egemen Erbayat  
For any questions or discussions, please reach out through erbayat@gwu.edu 
