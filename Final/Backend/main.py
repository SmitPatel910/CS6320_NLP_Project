# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
import json
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model 1: GPT-2 for recipe name generation
MODEL_PATH = "ft_gpt2_receipeNameGenerator/gpt2-recipes-ft/checkpoints"
USE_CPU = True
recipe_model = None
tokenizer = None

def load_model():
    """Load and initialize all required models."""
    global recipe_model, tokenizer
    logger.info("Loading Recipe Name Generation Model - FT-GPT2...")
    # Set device
    device = torch.device("cpu" if USE_CPU else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load recipe name generator model
        from ft_gpt2_receipeNameGenerator.model import load_model_and_tokenizer
        recipe_model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
        recipe_model.to(device)
        recipe_model.eval()
        # Add Green Success Icon
        logger.info(f"✅ Recipe Name Generation Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def generate_recipe_names(ingredients, tags):
    """Generate recipe names based on ingredients and tags."""
    try:
        from ft_gpt2_receipeNameGenerator.model import generate_recipes

        generated_recipes = generate_recipes(
            recipe_model,
            tokenizer,
            ingredients=ingredients,
            tags=tags,
            max_length=100,
            live_eval=True
        )
        
        return generated_recipes
    
    except Exception as e:
        logger.error(f"Error generating recipe names: {str(e)}")
        raise

def generate_recipe_instructions(recipe_name):
    """Generate recipe instructions using Google's Gemini model."""
    try:
        from rag_gemini_instructionGenerator.Chat_Chef import generate_recipe_steps
        response = generate_recipe_steps(recipe_name)
        
        result = {
            "title": recipe_name,
            "steps": response
        }
        return result
    
    except Exception as e:
        logger.error(f"Error generating recipe instructions: {str(e)}")
        # Provide fallback response if API fails
        return {
            "title": recipe_name,
            "steps": ["Sorry, I couldn't generate instructions at the moment. Please try again later."],
        }

@app.route('/api/generate-recipes', methods=['POST'])
def api_generate_recipes():
    """API endpoint to generate recipe names."""
    try:
        data = request.json
        ingredients = data.get('ingredients', [])
        tags = data.get('tags', [])
        
        print(f"Received ingredients: {ingredients}")
        print(f"Received tags: {tags}")

        if not ingredients:
            return jsonify({"error": "No ingredients provided"}), 400
        
        results = generate_recipe_names(ingredients, tags)
        print(f"Generated recipe names: {results}")
        return jsonify({"receipe_names": results})
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-instructions', methods=['POST'])
def api_generate_instructions():
    """API endpoint to generate recipe instructions."""
    try:
        data = request.json
        recipe_name = data.get('recipe_name', '')
        
        if not recipe_name:
            return jsonify({"error": "No recipe name provided"}), 400
        
        result = generate_recipe_instructions(recipe_name)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    """Serve the main application page."""
    with open('index.html', 'r') as file:
        return file.read()

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "models_loaded": recipe_model is not None})

if __name__ == "__main__":
    # Load models before starting the server
    load_model()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5100, debug=False)