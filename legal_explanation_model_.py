import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load CSV file
df = pd.read_csv("legal_data.csv")

# For now, add a dummy 'label' column (you should replace this with actual labels)
# For example, let's say we want to classify whether the suggestion is favorable (1) or not (0)
df['label'] = df['suggestion'].apply(
    lambda x: 1 if 'may be granted' in x.lower() or 'bail' in x.lower() else 0)

# Tokenizer and Model Initialization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

# Tokenization function


def tokenize_function(examples):
    return tokenizer(examples['suggestion'], truncation=True, padding='max_length', max_length=128)


# Convert pandas DataFrame to Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set the format to include the input_ids, attention_mask, and labels
tokenized_datasets = tokenized_datasets.with_format(
    type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("./legal_explanation_model")
