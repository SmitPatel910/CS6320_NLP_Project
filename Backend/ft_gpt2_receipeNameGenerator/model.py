# model.py
"""Load GPT-2 and extend embeddings for the <recipe> token."""

import torch
import re
from typing import List

def load_model_and_tokenizer(model_name: str = "gpt2"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ft_gpt2_receipeNameGenerator.dataset import RECIPE_TOKEN
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens = {"additional_special_tokens": [RECIPE_TOKEN]}

    print("Adding special tokens if missing...")
    added = tokenizer.add_special_tokens(special_tokens)
    print(f"Special Token ID for {RECIPE_TOKEN}:", tokenizer.convert_tokens_to_ids(RECIPE_TOKEN))
    print("Special tokens list:", tokenizer.all_special_tokens)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token (GPT-2 quirk)")

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if added:
        print(f"Resizing model embeddings to accommodate {added} new token(s)...")
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def clean_and_split(prediction: str) -> List[str]:
    """Clean and split generated text into recipe names."""
    from ft_gpt2_receipeNameGenerator.dataset import RECIPE_TOKEN
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    eos_token = tokenizer.eos_token  # <|endoftext|>

    # Token-based splitting if recipe tokens are present
    if prediction.count(RECIPE_TOKEN) >= 2:
        parts = prediction.split(RECIPE_TOKEN)
    else:
        parts = re.split(r"  +|,|\n", prediction)

    seen = set()
    results = []
    for part in parts:
        # Strip eos token if it's present inside the string
        part = part.replace(eos_token, "")
        name = part.strip().lower()
        if name and name != RECIPE_TOKEN.lower() and len(name.split()) >= 2 and name not in seen:
            seen.add(name)
            results.append(name)
        if len(results) == 3:
            break

    return results

def generate_recipes(
    model, 
    tokenizer, 
    ingredients: List[str], 
    tags: List[str], 
    max_length: int = 100,
    live_eval: bool = False
) -> List[str]:
    """
    Generate recipe names based on ingredients and tags for real-time inference.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use for encoding/decoding
        ingredients: List of ingredient names
        tags: List of tags
        max_length: Maximum length of generated sequence
        live_eval: Whether this is for real-time inference
        
    Returns:
        List of generated recipe names
    """
    from ft_gpt2_receipeNameGenerator.dataset import RECIPE_TOKEN
    from ft_gpt2_receipeNameGenerator.dataset import RecipeDataset

    # Create a dataset for encoding
    temp_dataset = RecipeDataset("", tokenizer.name_or_path, max_length, live_eval=True)
    
    # Build the prompt
    rec = {"ingredients": ingredients, "tags": tags}
    prompt = temp_dataset._build_prompt(rec) + f" {RECIPE_TOKEN} "
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            do_sample=not live_eval,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract the generated part
    prompt_length = inputs["input_ids"].shape[1]
    gen_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=False)
    
    # Process the generated text
    return clean_and_split(gen_text)

