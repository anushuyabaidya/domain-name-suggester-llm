"""
Created By: Anushuya Baidya
Date: 7/28/25
"""

import random

import pandas as pd
import torch

random.seed(42)

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

class DomainModel:
    """Fine-tuned model for domain generation"""
    def __init__(self, model_name="gpt2"):
        print(f"Loading {model_name}...")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        print("Working on device: ", self.device)
        self.model.to(self.device)

        self.blocked_words = ["adult", "porn", "gambling", "casino", "betting"]

    def prepare_training_data(self, training_df):
        """Convert training data to format the model can use"""
        texts = training_df['training_text'].tolist()
        def tokenize_function(examples):
            return self.tokenizer(
                examples,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )

        tokenized = tokenize_function(texts)

        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids']
        })

        print(f"Prepared {len(dataset)} training examples")
        return dataset

    def train_model(self, training_dataset, output_dir="../models/domain_model", epochs=3):
        """Train the model on domain generation task"""
        print(f"Starting training for {epochs} epochs...")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            learning_rate=5e-5
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        print(f"Training complete! Model saved to {output_dir}")
        return output_dir

    def is_safe_input(self, text):
        """Check if input is safe to process"""
        text_lower = text.lower()
        for word in self.blocked_words:
            if word in text_lower:
                return False
        return True

    def generate_domains(self, business_description):
        """Generate domain suggestions for a business"""
        if not self.is_safe_input(business_description):
            return {
                "domains": [],
                "status": "blocked",
                "message": "Request contains inappropriate content"
            }

        if not business_description.strip():
            return {
                "domains": [],
                "status": "error",
                "message": "Business description cannot be empty"
            }

        try:
            prompt = f"Business: {business_description} Domains:"

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            domains = self.extract_domains(generated_text)

            if domains:
                return {
                    "domains": domains,
                    "status": "success",
                    "message": f"Generated {len(domains)} domain suggestions"
                }
            else:
                return {
                    "domains": [],
                    "status": "error",
                    "message": "Could not generate valid domains"
                }

        except Exception as e:
            return {
                "domains": [],
                "status": "error",
                "message": f"Generation failed: {str(e)}"
            }

    def extract_domains(self, generated_text):
        """Extract domain names from generated text"""

        if "Domains:" in generated_text:
            domains_part = generated_text.split("Domains:")[-1].strip()
        else:
            domains_part = generated_text

        import re
        domain_pattern = r'([a-zA-Z0-9-]+\.(?:com|net|org|io|co))'
        domains = re.findall(domain_pattern, domains_part)

        clean_domains = []
        for domain in domains:
            clean_domain = domain.lower().strip()
            if (3 <= len(clean_domain.split('.')[0]) <= 25 and
                clean_domain not in clean_domains):
                clean_domains.append(clean_domain)

                if len(clean_domains) >= 3:
                    break

        return clean_domains

    def load_trained_model(self,model_path="../models_old/domain_model"):
        """Load a previously trained model"""
        print(f"Loading trained model from {model_path}...")
        model = DomainModel(model_path)
        print("Model loaded successfully!")
        return model

def train_domain_model(training_data_file="../data/training_data.csv"):
    """Complete training pipeline"""

    print("Domain Model Training Pipeline\n")

    print("Loading training data...")
    training_df = pd.read_csv(training_data_file)
    print(f"Loaded {len(training_df)} training examples")

    print("\nInitializing model...")
    model = DomainModel("gpt2")

    print("\nPreparing training data...")
    training_dataset = model.prepare_training_data(training_df)

    print("\nTraining model...")
    model_path = model.train_model(training_dataset, epochs=3)

    print("\nTesting trained model...")

    test_cases = [
        "organic coffee shop downtown",
        "innovative AI startup",
        "local yoga studio"
    ]

    for test_case in test_cases:
        result = model.generate_domains(test_case)
        print(f"Business: {test_case}")
        print(f"Domains: {result['domains']}")
        print(f"Status: {result['status']}")
        print("---")

    return model, model_path

if __name__ == "__main__":
    training_file_path = "../data/training_data.csv"
    model_output_path = "../models/domain_model"

    try:
        trained_model, model_path = train_domain_model(training_file_path)

        print("\nTraining Summary:")
        print(f"- Model type: GPT-2 fine-tuned")
        print(f"- Training examples: Check training_data.csv")
        print(f"- Model saved to: {model_path}")
        print(f"- Ready for evaluation!")

    except FileNotFoundError:
        print("training_data.csv not found!")
        print("Run the data generation script first to create training data.")

    except Exception as e:
        print(f"Training failed: {e}")
        print("Check your Python environment and GPU/CPU setup.")
