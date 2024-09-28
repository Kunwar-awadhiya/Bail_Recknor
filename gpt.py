from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

# Load and preprocess the dataset
data = pd.read_csv('legal_data.csv')
data = data[['offense_type', 'suggestion']].rename(
    columns={'offense_type': 'input_text', 'suggestion': 'target_text'})

# Convert to Dataset format
dataset = Dataset.from_pandas(data)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset


def tokenize_function(examples):
    inputs = tokenizer(
        examples['input_text'], truncation=True, padding='max_length', max_length=128)
    targets = tokenizer(
        examples['target_text'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = targets['input_ids']
    return inputs


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir='./results',
    num_train_epochs=3,
    logging_dir='./logs',
    evaluation_strategy="steps",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_gpt2')
tokenizer.save_pretrained('fine_tuned_gpt2')
