import torch
from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer
import yaml
from pykeen.datasets import WN18RR, FB15k237
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from functions import *


# Load the configuration file
with open('04_py_files/config_kg_nlm.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

device = torch.device("cuda")

# Get the desired model and training configuration
model_name = config['model']
tokenizer_name = config['tokenizer']
path = config['path']
config_parameters = config['training_arguments']

# Load train/test/val sets
train_triples = load_dict(path["train"])
test_triples = load_dict(path["test"])
val_triples = load_dict(path["val"])

# Load tokenizer and tokenize the input
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
train_tokenized = tokenizer(train_triples['triples'], padding=True)
test_tokenized = tokenizer(test_triples["triples"], padding=True)
val_tokenized = tokenizer(val_triples["triples"], padding=True)

# Turn input sequence into pytorch custom dataset format to prepare for NLM model
train_data = kg_nlm_nput(train_tokenized, train_triples["labels"])
test_data = kg_nlm_nput(test_tokenized, test_triples["labels"])
val_data = kg_nlm_nput(val_tokenized, val_triples["labels"])

# Clear cuda memory
del train_tokenized
del test_tokenized
del val_tokenized
del tokenizer
del train_triples
del test_triples
del val_triples


# Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Allow for backpropagation in all layers → set to true
for param in model.base_model.parameters():
    #param.requires_grad = False
    param.requires_grad = True

# Set training parameters from config file
training_args = TrainingArguments(**config_parameters)

# Initialize a Trainer construct with the selected model and specified training parameters
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data
)

# Train the model
trainer.train()

# Save trained model
trainer.save_model(path["trained_model"])

# Predict on test dataset and print metrics
output = trainer.predict(test_dataset=test_data)




