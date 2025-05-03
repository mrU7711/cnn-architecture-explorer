# CNN Architecture Explorer

A comparative analysis framework for evaluating and benchmarking popular CNN architectures (VGG, ResNet, DenseNet) through transfer learning on CIFAR-10.

## Overview

This project provides tools to load, fine-tune, and compare multiple pretrained CNN architectures. It measures accuracy, parameter count, computational cost (FLOPs), and inference speed to help understand architectural trade-offs.

## Project Structure

```
cnn-architecture-explorer/
├── src/
│   ├── architectures.py     # Model loading and modification
│   ├── transfer_learning.py # Fine-tuning pipeline
│   ├── benchmark.py         # Performance benchmarking
│   ├── dataset.py           # CIFAR-10 data loading
│   ├── analyzer.py          # Architecture analysis tools
│   ├── train.py             # Training pipeline
│   └── visualize.py         # Comparison visualizations
├── tests/
│   ├── test_architectures.py
│   └── test_benchmark.py
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/mrU7711/cnn-architecture-explorer.git
cd cnn-architecture-explorer
pip install -r requirements.txt
```

## Usage

```bash
python -m src.train --config config/config.yaml
```

## Architecture Comparison

| Architecture | Params (M) | Top-1 Acc | Inference (ms) |
|-------------|------------|-----------|----------------|
| VGG-16 | 138.4 | 92.1% | 12.3 |
| ResNet-18 | 11.7 | 93.5% | 4.2 |
| ResNet-50 | 25.6 | 94.2% | 8.7 |
| DenseNet-121 | 8.0 | 93.8% | 9.1 |

## License

MIT License
