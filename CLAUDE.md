# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a pipeline for training Google Gemma 3 1B language models from scratch on Dutch text data using the `BramVanroy/wikipedia_culturax_dutch` dataset from Hugging Face. The project is designed to be modular, simple to use, and optimized for fast iteration on an AWS g5.12xlarge instance with 4x A10G GPUs.

## Core Components

The project consists of three main scripts:

1. **preprocess.py**: Downloads and processes the Dutch Wikipedia dataset using Gemma 3 tokenizer
2. **training.py**: Trains a Gemma 3 1B model on the preprocessed data
3. **inference.py**: Generates text using the trained Gemma 3 model

## Prerequisites

### HuggingFace Authentication

**IMPORTANT**: The Gemma 3 1B model (`google/gemma-3-1b-pt`) is a gated model that requires authentication and license acceptance.

Before running any scripts, you must:

1. **Create a HuggingFace account** at https://huggingface.co/join
2. **Accept the Gemma license** at https://huggingface.co/google/gemma-3-1b-pt
3. **Create an access token** at https://huggingface.co/settings/tokens (with "Read" permissions)
4. **Login via CLI**:
   ```bash
   # Install huggingface-hub if not already installed
   pip install huggingface-hub
   
   # Login with your token
   huggingface-cli login
   # Or set environment variable
   export HF_TOKEN="your_token_here"
   ```

**Note**: All scripts will automatically use your HuggingFace credentials once logged in. If you encounter authentication errors, ensure you've completed all steps above.

## Common Commands

### Data Preprocessing

```bash
# Process a small test dataset (10k tokens) - good for testing
python preprocess.py --dataset-size 10k

# Process a small dataset (1M tokens)
python preprocess.py --dataset-size 1M

# Process a larger dataset (100M tokens)
python preprocess.py --dataset-size 100M --num-proc 16

# Process with custom parameters
python preprocess.py --dataset-size 10M --chunk-size 1024 --test-split 0.002 --output-dir custom_data
```

### Model Training

**IMPORTANT**: Always use `accelerate launch` for multi-GPU training to avoid CUDA out of memory errors.

```bash
# Train on test dataset (10k tokens) - good for testing
accelerate launch training.py --dataset-path processed_data/chunked_10k

# Train on 1M tokens
accelerate launch training.py --dataset-path processed_data/chunked_1M

# Train with wandb logging
accelerate launch training.py --dataset-path processed_data/chunked_10M --use-wandb

# Train with custom parameters (note smaller batch size for memory efficiency)
accelerate launch training.py --dataset-path processed_data/chunked_100M --epochs 1 --batch-size 2 --gradient-accumulation-steps 16
```

### Text Generation

```bash
# Generate text in interactive mode
python inference.py --model-path trained_models/gemma3-1b-dutch-latest/final_model --interactive

# Generate text with default prompts
python inference.py --model-path trained_models/gemma3-1b-dutch-latest/final_model

# Run benchmarking
python inference.py --model-path trained_models/gemma3-1b-dutch-latest/final_model --benchmark
```

## Architecture Details

### Data Processing Pipeline

The preprocessing script follows this workflow:
1. Load dataset from Hugging Face (configurable size: 10k, 1M, 10M, 100M, 1B, 10B tokens)
2. Create train/test split
3. Tokenize using Gemma 3 tokenizer (262K vocabulary)
4. Concatenate all texts and create fixed-size chunks (default: 1024 tokens)
5. Save processed dataset to disk

**Note**: The chunking uses a concatenation-based approach that combines all texts before splitting into chunks, which is more efficient and stable than per-document chunking.

### Training Pipeline

The training script implements:
1. Gemma 3 1B architecture with sliding window attention
2. Distributed training across available GPUs
3. Mixed-precision training with bfloat16
4. Learning rate scheduling optimized for Gemma
5. Integration with Weights & Biases (optional)

### Inference System

The inference script provides:
1. Interactive and batch text generation
2. Configurable generation parameters (temperature, top-k, top-p, repetition penalty)
3. Performance benchmarking optimized for Gemma 3

## Performance Expectations

- **Preprocessing** (on g5.12xlarge): 
  - 10B tokens: ~2-3 hours

- **Training** (on g5.12xlarge with 4x A10G GPUs):
  - 1B tokens: ~24-48 hours (depending on configuration)

- **Inference**:
  - Single GPU: ~15-25 tokens per second
  - Multi-GPU: ~40-80 tokens per second

## Troubleshooting

### Common Issues and Solutions

**CUDA out of memory errors during training:**
- **Issue**: `torch.OutOfMemoryError: CUDA out of memory`
- **Solution**: Use `accelerate launch` instead of running `python training.py` directly. This enables proper multi-GPU distributed training across all 4 A10G GPUs
- **Additional**: Reduce batch size to 2-4 and increase gradient accumulation steps to 8-16 for Gemma 3 1B
- **Setup**: Run `accelerate config` first to configure for 4-GPU training, or use the default config created automatically

**HuggingFace authentication errors:**
- **Issue**: `Repository not found` or `Access denied` errors when loading Gemma 3 model/tokenizer
- **Solution**: Ensure you've completed HuggingFace authentication steps above:
  1. Accept license at https://huggingface.co/google/gemma-3-1b-pt
  2. Run `huggingface-cli login` with a valid access token
  3. Verify login with `huggingface-cli whoami`
- **Alternative**: Set `HF_TOKEN` environment variable with your access token

**Tokenizer compatibility issues:**
- **Issue**: Vocabulary mismatch or encoding errors
- **Solution**: Ensure using `google/gemma-3-1b-pt` tokenizer consistently across preprocessing, training, and inference
- **Note**: Gemma 3 uses a 262K token vocabulary, much larger than standard models

**Model configuration errors:**
- **Issue**: `AttributeError` related to Gemma3TextConfig or Gemma3ForCausalLM
- **Solution**: Ensure transformers version >= 4.50.0 which includes Gemma 3 support
- **Update**: Run `pip install 'transformers>=4.50.0'` if needed

**Poor text generation quality:**
- **Issue**: Generated text contains random words or nonsensical output
- **Solution**: This is expected when training on very small datasets (like 10k tokens). Use larger datasets (1M+ tokens) for better results
- **Optimization**: Adjust temperature (0.7-0.9), top_p (0.9), and repetition_penalty (1.1) for better quality

**Accelerate version compatibility:**
- **Issue**: `ImportError: Using the 'Trainer' with 'PyTorch' requires 'accelerate>=0.26.0'`
- **Solution**: Run `pip install 'accelerate>=0.26.0'` to update the package

**Memory usage during inference:**
- **Issue**: High memory usage during text generation
- **Solution**: Use `torch_dtype=torch.bfloat16` and `device_map="auto"` for efficient memory management
- **Option**: Consider using smaller max_new_tokens or batch inference for large-scale generation

### Hardware-Specific Notes

**For AWS g5.12xlarge (4x A10G GPUs):**
- Recommended batch size: 2-4 per GPU
- Gradient accumulation: 8-16 steps
- Total effective batch size: 64-256 samples
- Memory usage: ~20-22GB per GPU during training

**For other GPU configurations:**
- Single GPU (24GB): batch_size=1, gradient_accumulation=32
- 2x GPUs: batch_size=2, gradient_accumulation=16
- 8x GPUs: batch_size=8, gradient_accumulation=4