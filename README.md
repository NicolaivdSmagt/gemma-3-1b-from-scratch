# Gemma 3 1B Dutch Language Model Training

A complete pipeline for training Google's Gemma 3 1B language model from scratch on Dutch text data. This project uses the `BramVanroy/wikipedia_culturax_dutch` dataset and is optimized for multi-GPU training on AWS cloud instances.

## Overview

This repository provides a streamlined approach to:
- Preprocess Dutch Wikipedia text data for language model training
- Train Gemma 3 1B models with configurable architectures
- Generate Dutch text using trained models
- Benchmark model performance and quality

## Key Features

- **Modern Architecture**: Uses Google's latest Gemma 3 1B model with sliding window attention
- **Scalable Training**: Distributed training across multiple GPUs using Hugging Face Accelerate
- **Flexible Data Processing**: Configurable dataset sizes from 10K to 10B tokens
- **Production Ready**: Includes checkpointing, monitoring, and inference capabilities

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Preprocessing

Process a small dataset for testing:
```bash
python preprocess.py --dataset-size 10k
```

Process larger datasets:
```bash
python preprocess.py --dataset-size 1M --num-proc 16
python preprocess.py --dataset-size 100M --num-proc 16
```

### Model Training

Train using accelerate for multi-GPU support:
```bash
# Small test run
accelerate launch training.py --dataset-path processed_data/chunked_10k

# Production training
accelerate launch training.py --dataset-path processed_data/chunked_1M --use-wandb
```

### Text Generation

Generate text with your trained model:
```bash
# Interactive mode
python inference.py --model-path trained_models/gemma3-1b-dutch-latest/final_model --interactive

# Batch generation
python inference.py --model-path trained_models/gemma3-1b-dutch-latest/final_model
```

## Architecture Details

### Gemma 3 1B Configuration
- **Parameters**: 1 billion parameters
- **Vocabulary**: 262,208 tokens (multilingual support)
- **Context Length**: 32,768 tokens
- **Architecture**: Transformer with sliding window attention
- **Precision**: bfloat16 for optimal performance

### Training Optimizations
- Sliding window attention for long sequences
- Gradient accumulation for large effective batch sizes
- Cosine learning rate scheduling
- Mixed precision training with bfloat16
- Distributed training across multiple GPUs

## Performance Expectations

### Hardware Requirements
- **Minimum**: 1x GPU with 24GB VRAM
- **Recommended**: 4x A10G GPUs (AWS g5.12xlarge)
- **Memory**: 32GB+ system RAM

### Training Times (4x A10G GPUs)
- **10K tokens**: ~5 minutes (testing)
- **1M tokens**: ~30 minutes
- **100M tokens**: ~8 hours
- **1B tokens**: ~24-48 hours

### Inference Speed
- **Single GPU**: 15-25 tokens/second
- **Multi-GPU**: 40-80 tokens/second (parallel generation)

## Project Structure

```
gemma-3-1b-from-scratch/
├── preprocess.py          # Data preprocessing pipeline
├── training.py            # Model training script
├── inference.py           # Text generation and inference
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── CLAUDE.md             # Development instructions
├── processed_data/       # Preprocessed datasets
├── trained_models/       # Saved model checkpoints
└── wandb/               # Training logs and metrics
```

## Dataset Information

Uses the `BramVanroy/wikipedia_culturax_dutch` dataset, which provides:
- High-quality Dutch Wikipedia content
- Multiple size configurations (10K to 10B tokens)
- Pre-cleaned and formatted text data
- Consistent tokenization across dataset sizes

## Advanced Usage

### Custom Training Configurations

```bash
# Large batch training with gradient accumulation
accelerate launch training.py \
  --dataset-path processed_data/chunked_100M \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-5

# Extended training with custom scheduling
accelerate launch training.py \
  --dataset-path processed_data/chunked_1B \
  --epochs 3 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01
```

### Generation Parameters

```bash
# Creative generation
python inference.py \
  --model-path trained_models/your-model \
  --temperature 1.0 \
  --top-p 0.95 \
  --repetition-penalty 1.1

# Focused generation
python inference.py \
  --model-path trained_models/your-model \
  --temperature 0.3 \
  --top-k 20 \
  --num-beams 5
```

## Monitoring and Logging

### Weights & Biases Integration

Enable experiment tracking:
```bash
accelerate launch training.py \
  --dataset-path processed_data/chunked_1M \
  --use-wandb \
  --wandb-project my-gemma-project
```

Track metrics:
- Training and validation loss
- Learning rate schedules
- Gradient norms
- Memory usage
- Training speed

## Contributing

This project is designed for research and educational purposes. Contributions welcome for:
- Performance optimizations
- Additional language support
- Novel training techniques
- Evaluation metrics

## License

This project uses the Gemma 3 model under Google's terms. Please review the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) before commercial use.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gemma3_dutch_training,
  title={Gemma 3 1B Dutch Language Model Training},
  author={Claude},
  year={2024},
  url={https://github.com/your-username/gemma-3-1b-from-scratch}
}
```