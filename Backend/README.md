# Recipe Name Generator

A fine-tuned GPT-2 model that generates creative recipe names based on ingredients and tags.

## Project Overview

This project uses a fine-tuned GPT-2 model to generate recipe names based on a list of ingredients and tags. The model has been trained on recipe data to understand the relationship between ingredients, tags, and recipe names.

## Features

- Generate creative recipe names from ingredients and tags
- Fine-tune on custom recipe datasets
- Real-time inference via interactive chat


### Training the Model

To fine-tune the model on your recipe dataset:

```bash
python run.py --train dataset/recipes_train_multi.jsonl \
              --val dataset/recipes_val_multi.jsonl \
              --model gpt2 \
              --output gpt2-recipes-ft \
              --epochs 5 \
              --batch 4
```

### Evaluating the Model

To evaluate the model on a validation set:

```bash
python run.py --eval_only \
              --train dataset/recipes_train_multi.jsonl \
              --val dataset/recipes_val_multi.jsonl \
              --model gpt2-recipes-ft
```

### Real-time Inference with Command Line

For real-time inference directly from the command line:

```bash
python run.py --live_eval \
              --model gpt2-recipes-ft \
              --ingredients "chicken, garlic, olive oil" \
              --tags "dinner, healthy, quick"
```

## Data Format

The dataset should be in JSONL format with each line containing a JSON object with:

- `ingredients`: List of strings representing ingredients
- `tags`: List of strings representing recipe categories/tags
- `names`: List of strings containing possible recipe names

Example:
```json
{"ingredients": ["chicken", "garlic", "olive oil"], "tags": ["dinner", "healthy"], "names": ["Garlic Roasted Chicken", "Simple Chicken Dinner"]}
```

## Model Details

The model uses GPT-2 with a special token `<recipe>` that helps separate multiple recipe suggestions. The generation process is optimized for both creative diversity and relevance to the provided ingredients and tags.

## Project Structure

- `accuracy.py`: Metrics for evaluating recipe name predictions
- `dataset.py`: Data loading and processing utilities
- `model.py`: Model definition and generation functions
- `run.py`: Main script for training, evaluation, and inference