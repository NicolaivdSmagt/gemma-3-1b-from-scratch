#!/usr/bin/env python3
"""
Gemma 3 1B Training: Inference Script

This script performs text generation with a trained Gemma 3 1B model.
It loads a model from a specified path and generates text based on provided prompts.

Features:
- Interactive or batch text generation
- Configurable generation parameters (temperature, top_k, top_p)
- Multiple prompt templates for testing Dutch text generation
- Performance benchmarking optimized for Gemma 3 architecture

Usage:
  python inference.py --model-path trained_models/gemma3-1b-dutch-latest/final_model

Author: Claude
"""

import os
import argparse
import time
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from huggingface_hub import whoami
from typing import List, Dict, Any, Optional

# Default Dutch prompts for testing
DEFAULT_PROMPTS = [
    "Nederland is een prachtig land waar",
    "De Nederlandse cultuur kenmerkt zich door",
    "In Amsterdam kun je",
    "Het weer in Nederland is",
    "De geschiedenis van Nederland toont",
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text with trained Gemma 3 1B model")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation (if not provided, default Dutch prompts will be used)"
    )
    
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Path to a file containing prompts (one per line)"
    )
    
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty to reduce repetitive text"
    )
    
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Number of beams for beam search (1 = no beam search)"
    )
    
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sequences to generate for each prompt"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save generated text (if not provided, text will be printed to console)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode, prompting for user input"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmarking on generation speed"
    )
    
    return parser.parse_args()

def get_prompts(args) -> List[str]:
    """Get prompts from various sources based on arguments."""
    if args.interactive:
        return []
    
    if args.prompt:
        return [args.prompt]
    
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    return DEFAULT_PROMPTS

def generate_text(
    model: Gemma3ForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    num_beams: int = 1,
    num_return_sequences: int = 1,
) -> List[str]:
    """Generate text from the model given a prompt."""
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    
    # Record start time for benchmarking
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Decode the generated text
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Calculate tokens per second
    tokens_generated = sum(len(output) - len(input_ids[0]) for output in outputs)
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    return generated_texts, generation_time, tokens_per_second

def run_interactive_mode(model: Gemma3ForCausalLM, tokenizer: AutoTokenizer, args):
    """Run the model in interactive mode, prompting for user input."""
    print("\n=== Gemma 3 1B Dutch Interactive Mode ===")
    print("Enter prompts to generate text. Type 'exit' to quit.")
    
    while True:
        # Get user input
        prompt = input("\nPrompt: ")
        
        # Check for exit command
        if prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive mode.")
            break
        
        # Skip empty prompts
        if not prompt.strip():
            continue
        
        # Generate text
        generated_texts, generation_time, tokens_per_second = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
        )
        
        # Print each generated text
        for i, text in enumerate(generated_texts):
            print(f"\nGenerated text {i+1}:")
            print(f"{text}")
        
        print(f"\nGeneration time: {generation_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")

def run_batch_mode(model: Gemma3ForCausalLM, tokenizer: AutoTokenizer, prompts: List[str], args):
    """Run the model in batch mode, generating text for all provided prompts."""
    all_results = []
    total_generation_time = 0
    total_tokens_per_second = 0
    
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
        
        # Generate text
        generated_texts, generation_time, tokens_per_second = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
        )
        
        # Store results
        result = {
            "prompt": prompt,
            "generated_texts": generated_texts,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
        }
        all_results.append(result)
        
        # Update totals for benchmarking
        total_generation_time += generation_time
        total_tokens_per_second += tokens_per_second
        
        # Print generated text
        for j, text in enumerate(generated_texts):
            print(f"\nGenerated text {j+1}:")
            print(f"{text}")
        
        print(f"Generation time: {generation_time:.2f}s ({tokens_per_second:.2f} tokens/sec)")
    
    # Print benchmark results if requested
    if args.benchmark and prompts:
        avg_generation_time = total_generation_time / len(prompts)
        avg_tokens_per_second = total_tokens_per_second / len(prompts)
        
        print("\n=== Benchmark Results ===")
        print(f"Total prompts: {len(prompts)}")
        print(f"Total generation time: {total_generation_time:.2f}s")
        print(f"Average generation time per prompt: {avg_generation_time:.2f}s")
        print(f"Average tokens per second: {avg_tokens_per_second:.2f}")
    
    # Save results to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(f"Prompt: {result['prompt']}\n\n")
                for i, text in enumerate(result['generated_texts']):
                    f.write(f"Generated text {i+1}:\n{text}\n\n")
                f.write(f"Generation time: {result['generation_time']:.2f}s")
                f.write(f" ({result['tokens_per_second']:.2f} tokens/sec)\n")
                f.write("-" * 50 + "\n\n")
        
        print(f"\nResults saved to {args.output_file}")

def main():
    """Main inference function."""
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
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = Gemma3ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Parameters: {model.num_parameters():,}")
    
    # Get prompts
    prompts = get_prompts(args)
    
    # Run in interactive or batch mode
    if args.interactive:
        run_interactive_mode(model, tokenizer, args)
    else:
        run_batch_mode(model, tokenizer, prompts, args)

if __name__ == "__main__":
    main()