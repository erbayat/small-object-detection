#!/bin/bash

# Array of model names
models=("yolo11n" "yolo11s" "yolo11m" "yolo11l" "yolo11x" "yolo11n-visdrone" "yolo11s-visdrone" "yolo11m-visdrone" "yolo11l-visdrone" "yolo11x-visdrone")

# Array of is_sahi values
is_sahi_values=(False True)

# Iterate over each model and is_sahi combination
for model in "${models[@]}"; do
    for is_sahi in "${is_sahi_values[@]}"; do
        echo "Running for model: $model with is_sahi: $is_sahi"
        python ./src/eval_models.py --model_name "$model" --is_sahi "$is_sahi"
    done
done
