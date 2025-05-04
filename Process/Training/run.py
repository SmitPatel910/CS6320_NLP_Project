# Standard library imports
from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Third-party library imports
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

# Local/application imports
from dataset import RecipeDataset, RECIPE_TOKEN
from model import load_model_and_tokenizer, clean_and_split
from accuracy import compute_soft_accuracy

# Configure logging and warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

def _maybe_relaunch_ddp(use_cpu):
    """
    Conditionally relaunch with distributed data parallel based on GPU availability and use_cpu flag.
    
    Args:
        use_cpu: If True, force CPU usage even if GPUs are available
    """
    if use_cpu or torch.cuda.device_count() <= 1:
        return
    already_launched = "LOCAL_RANK" in os.environ
    if already_launched:
        return
    cmd = [
        sys.executable,
        "-m", "torch.distributed.run",
        "--nproc_per_node", str(torch.cuda.device_count()),
        sys.argv[0], *sys.argv[1:],
    ]
    print(f"Detected {torch.cuda.device_count()} GPUs → relaunching with DDP:\n    {' '.join(cmd)}\n")
    os.execvp(cmd[0], cmd)

def _parse_args():
    p = argparse.ArgumentParser(description="Recipe Name Generator - Training and Evaluation")
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only (no training)")
    p.add_argument("--train", help="Path to training data")
    p.add_argument("--val", help="Path to validation data")
    p.add_argument("--model", default="gpt2", help="Model name or path")
    p.add_argument("--output", default="gpt2-recipes-ft", help="Output directory")
    p.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=4, help="Batch size per device")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    p.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPUs are available")
    return p.parse_args()

def _build_trainer(args):
    # Set device based on CPU flag
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model_and_tokenizer(args.model)
    
    # Move model to the appropriate device
    model = model.to(device)
    
    ds_train = RecipeDataset(args.train, tok.name_or_path, args.max_len)
    ds_val = RecipeDataset(args.val, tok.name_or_path, args.max_len)

    if len(ds_train.records) > 0:
        rec = ds_train.records[0]
        prompt = ds_train._build_prompt(rec)
        recipe_token_id = tok.convert_tokens_to_ids(RECIPE_TOKEN)
        print(f"Debug: prompt length={len(prompt)}, recipe_token_id={recipe_token_id}")

    # Calculate steps per epoch
    steps_per_epoch = len(ds_train) // args.batch
    if not args.cpu and torch.cuda.device_count() > 1:
        steps_per_epoch = steps_per_epoch // torch.cuda.device_count()

    targs = TrainingArguments(
        output_dir=args.output,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        logging_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
        fp16=args.fp16 and not args.cpu,
        report_to="none",
        save_total_limit=5,
        logging_dir=str(Path(args.output) / "logs"),
        logging_strategy="steps",
        disable_tqdm=False,
        # Force CPU if requested
        no_cuda=args.cpu,
    )

    return Trainer(
        model=model,
        args=targs,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=None,  # Default collator
        tokenizer=tok,        
    )

def _write_epoch_csv(trainer: Trainer, csv_path: Path):
    by_epoch = defaultdict(dict)
    for entry in trainer.state.log_history:
        ep = entry.get("epoch")
        if ep is None:
            continue
        if "loss" in entry:
            by_epoch[ep]["train_loss"] = entry["loss"]
        if "eval_loss" in entry:
            by_epoch[ep]["eval_loss"] = entry["eval_loss"]
        
    fieldnames = ["epoch", "train_loss", "eval_loss"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in sorted(by_epoch):
            writer.writerow({"epoch": ep, **by_epoch[ep]})

def eval_process(model, tokenizer, val_ds, use_cpu=False):
    """
    Evaluate model on validation dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer to use
        val_ds: Validation dataset
        use_cpu: If True, force CPU usage even if GPUs are available
        
    Returns:
        Tuple of (predictions list, accuracy value)
    """
    preds = []
    acc_val = 0
    
    # Set appropriate device
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    for rec in tqdm(val_ds.records, desc="Evaluating"):
        prompt = val_ds._build_prompt(rec) + f" {RECIPE_TOKEN} "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=100,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Extract generated IDs after the prompt
            gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            prediction = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            split_preds = clean_and_split(prediction)
            accuracy = compute_soft_accuracy(split_preds, rec["names"])
            acc_val += accuracy

            preds.append({
                "prompt": prompt,
                "prediction": prediction,
                "gold": ", ".join(rec["names"]),
                "accuracy": accuracy,
            })
    return preds, acc_val
    
def main():
    args = _parse_args()
    set_seed(args.seed)
    
    # Show device information
    if args.cpu:
        print("Forcing CPU usage as requested (--cpu flag)")
    else:
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_count > 0 else []
        print(f"Using {device} device{'s' if gpu_count > 1 else ''}: {', '.join(gpu_names) if gpu_names else 'CPU only'}")
        
    # Check if train and val paths are provided for training/regular eval
    if not args.train or not args.val:
        print("Error: --train and --val parameters are required for training or evaluation")
        sys.exit(1)
    
    # Regular training/evaluation flow
    trainer = _build_trainer(args)

    # Training
    if not args.eval_only:
        if trainer.is_world_process_zero():
            print("starting fine-tuning …")
        trainer.train()
        if trainer.is_world_process_zero():
            trainer.save_model(args.output)
            trainer.tokenizer.save_pretrained(args.output)
            _write_epoch_csv(trainer, Path(args.output) / "training_metrics.csv")

    # Evaluation
    if trainer.is_world_process_zero():
        print(f"Starting evaluation using model: {args.model}")
        model = trainer.model.eval()
        tokenizer = trainer.tokenizer
        val_ds = trainer.eval_dataset

        # Evaluate the model
        print("evaluating model …")
        preds, eval_accuracy = eval_process(model, tokenizer, val_ds, use_cpu=args.cpu)
        total_accuracy = eval_accuracy / len(val_ds.records)
        print(f"Evaluation accuracy: {total_accuracy:.2%}")

        df = pd.DataFrame(preds)
        df.to_csv(Path(args.output) / "eval_pred.csv", index=False)
        print(f"Evaluation CSV saved → {(Path(args.output) / 'eval_pred.csv').resolve()}")
        print(f"Total predictions made: {len(preds)}")
        print(f"Total accuracy: {total_accuracy:.2%}")
        print("Evaluation completed.")

if __name__ == "__main__":
    _maybe_relaunch_ddp(_parse_args().cpu)
    main()
