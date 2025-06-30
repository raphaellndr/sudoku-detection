# Sudoku Digit Recognition

A modular and configurable training system for digit classification, specifically designed for Sudoku digit recognition with optional MNIST pre-training.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for data, models, training, and CLI
- **Transfer Learning**: Support for MNIST pre-training and fine-tuning on Sudoku datasets
- **Poetry Integration**: Modern Python dependency management
- **Configurable Training**: Flexible configuration system for all training parameters
- **Multiple Training Modes**: Train from scratch, fine-tune, or use pre-trained models
- **Data Augmentation**: Customizable augmentation strategies for different datasets

## Installation

### Prerequisites
- Python 3.12+
- Poetry (install from [python-poetry.org](https://python-poetry.org/docs/#installation))

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd sudoku-digit-recognition

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

### Command Line Interface

The package provides a CLI tool accessible via `sudoku-train`:

#### Train a Sudoku Model
```bash
# Train from scratch
sudoku-train train --data-dir /path/to/sudoku/dataset

# Train with MNIST pre-training
sudoku-train train --data-dir /path/to/sudoku/dataset --use-mnist-pretraining

# Fine-tune from existing model
sudoku-train train --data-dir /path/to/sudoku/dataset --pretrained-model /path/to/model.keras

# Custom parameters
sudoku-train train \
    --data-dir /path/to/sudoku/dataset \
    --batch-size 64 \
    --epochs 150 \
    --learning-rate 0.0005 \
    --output-dir ./models
```

#### Train MNIST Only
```bash
sudoku-train train-mnist --output-dir ./models --epochs 50
```

#### Evaluate a Model
```bash
sudoku-train evaluate --model-path ./