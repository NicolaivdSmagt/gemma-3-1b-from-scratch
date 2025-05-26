#!/usr/bin/env python3
"""
Gemma 3 1B Training: Training Script

This script trains a Gemma 3 1B model on preprocessed Dutch text data.
It loads the processed dataset, initializes a Gemma 3 1B model, and trains it
using Hugging Face's Trainer API with distributed training on multiple GPUs.

Features:
- Gemma 3 1B architecture with sliding window attention
- Distributed training across multiple GPUs
- Learning rate scheduling optimized for Gemma architecture
- Gradient accumulation for large batch training
- Checkpointing and model saving
- Optional wandb integration for monitoring

Usage:
  accelerate launch training.py --dataset-path processed_data/chunked_1M

Author: Claude
"""

import os
import argparse
import math
import time
from datasets import load_from_disk
from huggingface_hub import whoami
from transformers import (
    AutoTokenizer,
    Gemma3TextConfig,
    Gemma3ForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Optional: Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Gemma 3 1B on Dutch text")
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to preprocessed dataset"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trained_models",
        help="Directory to save trained model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per GPU/CPU for training"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before backward pass"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (lower for Gemma)"
    )
    
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Proportion of training for learning rate warmup"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay to apply"
    )
    
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="gemma3-1b-dutch",
        help="Weights & Biases project name"
    )
    
    parser.add_argument(
        "--save-steps",
        type=int,
        default=2000,
        help="Save checkpoint every X updates steps"
    )
    
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate model every X updates steps"
    )
    
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Limit the total amount of checkpoints"
    )
    
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Check HuggingFace authentication
    try:
        print("Checking HuggingFace authentication...")
        user_info = whoami()
        print(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        print("❌ HuggingFace authentication required!")
        print("Please run: huggingface-cli login")
        print("Or set HF_TOKEN environment variable")
        print("And accept the license at: https://huggingface.co/google/gemma-3-1b-pt")
        raise e
    
    # Create output directory
    model_output_dir = os.path.join(
        args.output_dir, 
        f"gemma3-1b-dutch-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize wandb if enabled
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed. Proceeding without wandb logging.")
        else:
            wandb.init(
                project=args.wandb_project,
                name=f"gemma3-1b-dutch",
                config=vars(args)
            )
    
    # Load preprocessed dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset loaded. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")
    
    # Initialize tokenizer
    print("Loading Gemma 3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer)}")
    
    # Initialize model with Gemma 3 1B configuration
    print("Initializing Gemma 3 1B model...")
    config = Gemma3TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        max_position_embeddings=32768,  # 32K context for 1B model
        sliding_window=4096,
        rope_theta=1000000.0,
        rms_norm_eps=1e-06,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Gemma3ForCausalLM(config)
    print(f"Model initialized with {model.num_parameters():,} parameters")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up training arguments optimized for Gemma 3
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        logging_dir=os.path.join(model_output_dir, "logs"),
        logging_steps=100,
        report_to="wandb" if args.use_wandb and WANDB_AVAILABLE else "none",
        # Use bfloat16 for Gemma 3 (recommended over fp16)
        bf16=True,
        # Enable distributed training
        ddp_find_unused_parameters=False,
        # Memory optimizations
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    # Save final model
    final_model_path = os.path.join(model_output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Run final evaluation
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    eval_perplexity = math.exp(eval_results["eval_loss"])
    print(f"Final perplexity: {eval_perplexity:.2f}")
    
    # Log final metrics to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final_perplexity": eval_perplexity,
            "training_time_hours": training_time/3600
        })
        wandb.finish()

if __name__ == "__main__":
    main()