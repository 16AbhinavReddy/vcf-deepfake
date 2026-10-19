# VCF Deepfake Detection

A video deepfake detection project for CS725: Foundations of Machine Learning.

## Overview

This project investigates deepfake detection across videos with different compression levels. It uses a shared R3D-18 video backbone to extract features and predict whether a video is real or fake.

The training objective combines binary classification loss with a cosine-based consistency loss across compression variants.

## Dataset

The VCF dataset is stored locally and is not included in this repository.

Expected local structure:

    vcf/
      vcf/
        raw/
        c23/
        c40/

See the dataset's official documentation for its download instructions, label definitions, and source-video identifiers.

## Project Structure

    src/          Model, dataset, training, and evaluation code
    checkpoints/   Locally saved model checkpoints (not tracked by Git)
    vcf/           Local dataset (not tracked by Git)

## Environment

The project was developed using Python 3.12, PyTorch, torchvision, and OpenCV.

Activate your virtual environment before running the scripts:

    source .venv312/bin/activate

## Training and Evaluation

Training and evaluation commands depend on the configuration in `src/train.py` and `src/evaluate.py`.

Inspect those files for the current settings before running experiments.

## Experiments

The planned experiments compare classification-only training against classification with cosine consistency loss, and evaluate different learning rates, loss weights, and training durations.

## Disclaimer

This is an academic machine-learning project. Results should be evaluated using appropriate class-wise metrics and a source-independent data split.
