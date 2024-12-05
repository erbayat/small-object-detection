#!/bin/bash

# Array of model names
models=("yolo11n" "yolo11s" "yolo11m" "yolo11l" "yolo11x" "yolo11n-visdrone" "yolo11s-visdrone" "yolo11m-visdrone" "yolo11l-visdrone" "yolo11x-visdrone")


# Iterate over each model and is_sahi combination
for model in "${models[@]}"; do
    echo "Running for model: $model without sahi"
    python3 ./src/eval_models.py --model_name $model
    echo "Running for model: $model with sahi"
    python3 ./src/eval_models.py --model_name $model --is_sahi True
done
