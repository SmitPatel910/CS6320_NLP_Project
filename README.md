# Food Q&A: AI-Powered Recipe Generator
<div align="center">
  <img src="Process/Dataset/food.png" alt="Food Q&A Banner" width="200"/>
</div>
<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/Model-GPT--2-brightgreen" alt="GPT-2">
  <img src="https://img.shields.io/badge/Model-Google%20Gemini-blue" alt="Gemini">
</p>

## 🍽️ Overview

**Food Q&A** is an intelligent assistant that helps users design recipes based on available ingredients and dietary preferences. Our dual-model approach combines the power of a fine-tuned GPT-2 model for recipe name generation with a RAG-based Google Gemini AI system for detailed cooking instructions.

[**Watch our demo video here**](#) (Video coming soon)

## 🧪 Hypothesis & Learning Motivation

Our project explores two key hypotheses:

1. **Pre-trained language models can be specialized for culinary creativity**: We hypothesized that fine-tuning GPT-2, a powerful language model, on a large recipe dataset would produce a system capable of generating contextually appropriate and creative recipe names based on ingredients and tags that outperforms general-purpose language models.

2. **RAG enhances domain-specific instruction generation**: We proposed that Retrieval-Augmented Generation (RAG) with state-of-the-art generative models would provide more accurate, detailed, and contextually relevant cooking instructions than models without access to specialized knowledge retrieval.

These hypotheses drove our dual-system approach, leveraging the complementary strengths of fine-tuning and knowledge retrieval to create a comprehensive culinary assistant.

## 🔍 Methodology

### 1. Fine-tuned GPT-2 Recipe Name Generator

We finetuned GPT-2 model to suggest appropriate recipe names based on ingredients and tags:

- **Base Model**: Started with the pre-trained GPT-2.0 model
- **Dataset**: Utilized Recipe.CSV dataset containing 231,637 recipes
- **Data Split**: 80% training, 10% validation, 10% testing
- **Input Design**: Structured prompt format: `Ingredients: [list]\nTags: [list]\nRecipe name:`
- **Output Design**: Recipe name suggestions separated by `<recipe>` special tokens
- **Training Parameters**:
  - Hardware: NVIDIA RTX A6000
  - Epochs: 5
  - Batch size: 4
  - Learning rate: 5e-5
- **Model Performance**:
  - Validation Accuracy: 65.24%
  - Test Accuracy: 65.38%

**Technical Enhancements:**
- Implemented token-type embeddings to differentiate between ingredients, tags, and recipe names
- Applied dynamic masking during training to improve generalization
- Used focal loss to handle class imbalance in recipe categories
- Implemented beam search decoding during inference for diverse recipe suggestions
- Applied gradient accumulation to efficiently handle larger effective batch sizes

### 2. RAG-based Google Gemini Instruction Generator

Our second component leverages the Retrieval-Augmented Generation approach with Google's Gemini model:

- **Knowledge Base**: Created a specialized database of cooking instructions and techniques
- **Embedding System**: Vector embeddings stored in Faiss for efficient retrieval
- **Batch Ingestion**: Efficient batch upload process to handle large datasets
- **Query Processing**:
  1. Embed user's recipe name query
  2. Retrieve similar documents from Faiss
  3. Generate context-aware prompts for Gemini
  4. Deliver detailed, step-by-step cooking instructions

**Technical Enhancements:**
- Implemented semantic chunking to preserve recipe context during RAG
- Applied hybrid search combining keyword and semantic matching
- Developed query rewriting techniques to handle ambiguous culinary terminology
- Built a context window optimization system to maximize relevant information
- Created a response validation system for factual accuracy

## 🏗️ Project Structure

```
CS6320_NLP_Project
.
├── Backend
│   ├── ft_gpt2_receipeNameGenerator
│   │   ├── dataset.py
│   │   ├── gpt2-recipes-ft
│   │   │   └── checkpoints
│   │   ├── model.py
│   │   └── run.py
│   ├── main.py
│   ├── rag_gemini_instructionGenerator
│   │   ├── Chat_Chef.py
│   │   └── saved_embeddings.pkl
│   └── README.md
├── Frontend
│   └── index.html
├── LICENSE
├── Process
│   ├── Dataset
│   │   ├── RAW_recipes.csv
│   │   ├── README.md
│   ├── Dataset_Preprocess
│   │   ├── ft_gpt_2
│   │   │   ├── build_multi_targets.py
│   │   │   └── script.py
│   │   └── RAG_GoogleGemini
│   │       └── main.ipynb
│   └── Training
│       ├── accuracy.py
│       ├── dataset.py
│       ├── model.py
│       └── run.py
└── README.md
└── requirements.txt

```

## 🚀 How to Execute the Project

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key

### Setup and Installation

1. **Clone the repository**
   ```bash
   https://github.com/SmitPatel910/CS6320_NLP_Project.git
   cd CS6320_NLP_Project
   ```

2. **Create environment file**
   ```bash
   echo "KEY=your_gemini_api_key" > .env
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights**
   - Download the fine-tuned GPT-2 model [weights](https://drive.google.com/drive/folders/1CaNBXDmeAV1jml8u2GAGYPjRvHT-1cpA?usp=sharing)
   - Place them in `Backend/ft_gpt2_receipeNameGenerator/gpt2-recipes-ft/checkpoints/`

### Running the Application

1. **Start the backend server**
   ```bash
   cd Backend
   python main.py
   ```

2. **Launch the frontend**
   - Navigate to the `Frontend` directory
   - Open `index.html` in your browser

### Using the Application

Food Q&A offers two primary modes:

#### Mode 1: Recipe Generator (Fine-tuned GPT-2)
- Select dietary tags from the provided options
- Enter your available ingredients (comma-separated)
- Get creative recipe name suggestions

#### Mode 2: Ask Chef (RAG-based Gemini)
- Enter a recipe name
- Receive detailed, step-by-step cooking instructions

## 🔄 Reproducing the Dataset and Model Weights

### Dataset Acquisition
1. Download the raw dataset from [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv)
2. Place the downloaded file in `Process/Dataset/`

### Data Preprocessing
1. **For GPT-2 Fine-tuning**
   ```bash
   cd Process/Dataset_Preprocess/ft_gpt_2/
   python script.py
   python build_multi_targets.py
   ```

2. **For RAG System**
   ```bash
   cd Process/Dataset_Preprocess/RAG_GoogleGemini/
   jupyter notebook main.ipynb
   ```
   - Run all cells to generate `saved_embeddings.pkl`

### Model Training
```bash
cd Process/Training
python run.py --train <train_dataset_path> --val <val_dataset_path> --model gpt2 --output <output_directory> --epochs 5
```

### Model Evaluation
```bash
python run.py --eval_only --train <train_dataset_path> --val <val_dataset_path> --model <finetuned_model_checkpoints>
```

## 🔬 Results and Evaluation
### Recipe Name Generator (GPT-2)
- **BLEU Accuracy**: 65.24% on validation set
- **BLEU Accuracy**: 65.38% on test set

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
