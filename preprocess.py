#!/usr/bin/env python3
"""
Gemma 3 1B Training: Preprocessing Script

This script processes the BramVanroy/wikipedia_culturax_dutch dataset for Gemma 3 1B training.
It downloads the dataset, tokenizes the text using Gemma 3's tokenizer, and creates training 
chunks of fixed length optimized for the Gemma 3 architecture.

Features:
- Configurable dataset size (1M, 10M, 100M, 1B, 10B tokens)
- Tokenizes text using Gemma 3 tokenizer with 262K vocabulary
- Creates fixed-length chunks for efficient training
- Saves processed dataset to disk for training

Usage:
  python preprocess.py --dataset-size 1M --chunk-size 1024 --test-split 0.002

Author: Claude
"""

import os
import argparse
from itertools import chain
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess Dutch Wikipedia dataset for Gemma 3 1B training")

    parser.add_argument(
        "--dataset-size",
        type=str,
        default="1M",
        choices=["10k", "1M", "10M", "100M", "1B", "10B"],
        help="Size of the dataset to use (10k, 1M, 10M, 100M, 1B, 10B tokens)"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Size of text chunks for training"
    )

    parser.add_argument(
        "--test-split",
        type=float,
        default=0.002,
        help="Fraction of data to use for testing"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_data",
        help="Directory to save processed data"
    )

    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for dataset processing"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for dataset processing"
    )

    return parser.parse_args()

def main():
    """Main preprocessing function."""
    args = parse_args()

    start_time = time.time()
    print(f"Starting preprocessing with dataset size: {args.dataset_size}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set paths for processed data
    tokenized_path = os.path.join(args.output_dir, f"tokenized_{args.dataset_size}")
    chunked_path = os.path.join(args.output_dir, f"chunked_{args.dataset_size}")

    # Check if processed data already exists
    if os.path.exists(chunked_path):
        print(f"Processed data already exists at {chunked_path}. Skipping preprocessing.")
        return

    # Step 1: Load the dataset
    print(f"Loading Wikipedia Dutch dataset ({args.dataset_size} tokens)...")
    dataset = load_dataset("BramVanroy/wikipedia_culturax_dutch", args.dataset_size, trust_remote_code=True)
    print(f"Dataset loaded. Structure: {dataset}")

    # Step 2: Create train/test split
    dataset = dataset['train'].train_test_split(test_size=args.test_split)
    print(f"Dataset split created. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    # Step 3: Load Gemma 3 tokenizer
    print("Loading Gemma 3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}")

    # Step 4: Tokenize function
    def tokenize_function(example):
        return tokenizer(text=example["text"])

    # Step 5: Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_ds = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=['text']
    )
    print(f"Dataset tokenized. Number of samples: {len(tokenized_ds['train'])}")

    # Step 6: Concatenate all texts and create chunks
    print(f"Creating chunks of size {args.chunk_size}...")
    
    def group_texts(examples):
        """Concatenate all texts and create chunks of fixed size."""
        # Concatenate all the texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= args.chunk_size:
            total_length = (total_length // args.chunk_size) * args.chunk_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.chunk_size] for i in range(0, total_length, args.chunk_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Step 7: Create chunks by concatenating and splitting
    chunked_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=args.num_proc,
    )
    print(f"Chunks created. Train: {len(chunked_ds['train'])}, Test: {len(chunked_ds['test'])}")

    # Step 8: Save processed dataset
    print(f"Saving processed dataset to {chunked_path}...")
    chunked_ds.save_to_disk(chunked_path)

    # Print statistics
    elapsed_time = time.time() - start_time
    print("\n=== Preprocessing Statistics ===")
    print(f"Dataset size: {args.dataset_size} tokens")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Vocabulary size: {len(tokenizer)} tokens")
    print(f"Train samples: {len(chunked_ds['train'])}")
    print(f"Test samples: {len(chunked_ds['test'])}")
    print(f"Processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Processed data saved to: {chunked_path}")
    print("================================\n")

if __name__ == "__main__":
    main()