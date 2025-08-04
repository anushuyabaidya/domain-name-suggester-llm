# Domain Name Generator

This project fine-tunes a GPT-2 model to generate domain name suggestions for businesses. It includes data generation, model training, evaluation, and API.

## Setup requirements

**Using conda environment:**
```bash
# Create conda environment
conda create -n domain_env python=3.9
conda activate domain_env

# Install requirements
pip install -r requirements.txt
```

## What's in this project

### Main notebook
- `Domain_suggester.ipynb` - The complete pipeline in one notebook. Run this if you want to see everything working together.

### Source code (`src/` folder)
- `data_generator.py` - Creates synthetic business descriptions and matching domains for training
- `domain_trainer.py` - Handles model training and domain generation  
- `evaluator.py` - Tests model performance without needing expensive APIs
- `model_iterator.py` - Compares different model versions and shows improvements
- `domain_api.py` - Simple Flask API that serves domain suggestions

### Test folder
- `test_api.py` - Tests the API with different business descriptions

### Data and results
- `data/` - Training datasets (original and improved versions)
- `models/` - Saved model checkpoints (v1, v2, etc.)
- `results/` - Evaluation results and comparison metrics

## How to run this

### Option 1: Quick start with notebook
Just open `Domain_suggester.ipynb` and run all cells. This will:
1. Generate 1000 training examples
2. Train a GPT-2 model 
3. Evaluate the model
4. Create an improved version
5. Compare both models

### Option 2: Run individual components

**Generate training data:**
```bash
python src/data_generator.py
```

**Train the model:**
```bash
python src/domain_trainer.py
```

**Evaluate the model:**
```bash
python src/evaluator.py
```

**Improve the model:**
```bash
python src/model_iterator.py
```

**Start the API:**
```bash
python src/domain_api.py
```

**Test the API:**
```bash
python test/test_api.py
```

## What the model does

**Input:** Business description like "organic coffee shop downtown"  
**Output:** Domain suggestions like ["organicbeans.com", "coffeeshop.net", "localcafe.org"]

**Safety:** Blocks inappropriate content automatically  
**Quality:** Scores domains based on length, relevance, and memorability

## Project structure
```
domain_generator/
├── Domain_suggester.ipynb    # Complete pipeline notebook
├── src/
│   ├── data_generator.py     # Creates training data
│   ├── domain_trainer.py     # Model training
│   ├── evaluator.py         # Model evaluation  
│   ├── model_iterator.py    # Model improvement
│   └── domain_api.py        # API server
├── test/
│   └── test_api.py          # API testing
├── data/
│   ├── training_data.csv    # Original training data
│   └── training_data_v2.csv # Improved training data
├── models/
│   ├── domain_model/        # Baseline model
│   └── domain_model_v2/     # Improved model
├── results/
│   ├── model_evaluation_results.csv
│   ├── detailed_model_comparison.csv
│   └── edge_case_test_results.csv
└── requirements.txt         # Package dependencies
```

## API usage

Start the API:
```bash
python src/domain_api.py
```
##Response
```
{
  "suggestions": [
    {"domain": "coffeeshop.com", "confidence": 0.90},
    {"domain": "cafe.net", "confidence": 0.85}
  ],
  "status": "success"
}
```





