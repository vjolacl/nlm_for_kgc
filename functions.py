from pykeen.pipeline import pipeline
from pykeen.datasets import Nations
from pykeen.models import TransE
import torch
from typing import List
import pykeen.nn
from pykeen.models import predict
from pykeen.evaluation import RankBasedEvaluator
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from pykeen.nn.init import PretrainedInitializer

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from wordsegment import load, segment

def get_data():

    # Store dataset into separate training and test data
    dataset = Nations()
    dataset_train = dataset.training
    dataset_test = dataset.testing

    # Store the entity-to-id and relation-to-id relationship in separate dictionaries
    entities_to_ids = dataset.entity_to_id
    relations_to_ids = dataset.relation_to_id

    return entities_to_ids, relations_to_ids


#---------- BERT embeddings ----------

def generate_BERT_entity_embeddings(entities_to_ids, relations_to_ids):

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # BERT input: get KG entities/relations into separate lists, preserving the sequence in which they are extracted
    df_entities = pd.DataFrame(entities_to_ids.items(), columns= ["entity", "index"])
    kg_entities = list(entities_to_ids.keys())

    df_relations = pd.DataFrame(relations_to_ids.items(), columns=["relation", "index"])
    kg_relations = list(relations_to_ids.keys())


    # Tokenize the KG entities
    entities_to_tokens = tokenizer.batch_encode_plus(kg_entities, padding = True,return_tensors='pt')['input_ids']
    relations_to_tokens = tokenizer(kg_relations, padding=True, return_tensors='pt')['input_ids']

    # Map the token ID to entity
    df_entities["token ID"] = entities_to_tokens.tolist()

    # Load the BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval() # Question nr.1: What is the difference in putting model in evaluation mode or not?

    # Understanding the output
    output = model(entities_to_tokens)
    hidden_states = output["hidden_states"]

    return hidden_states, df_entities

def concat_hidden_states(hidden_states):
    # Concat last 4 hidden states for token 0 = CLS token, cls_embdd = hidden_states[:][:][0]
    embdd_concatenated = []

    # For each text sequence...
    for index in range(len(hidden_states[0])):
        # Concatenate the vectors from the last four layers. Each layer vector is 768 values, so `cat_vec` is length 4*768.
        cat_vec = torch.cat((hidden_states[-1][index][0], hidden_states[-2][index][0],
                             hidden_states[-3][index][0], hidden_states[-4][index][0]), dim=0)
        embdd_concatenated.append(cat_vec)

    # Convert output to 2D tensor
    embdd_tensor = torch.stack(embdd_concatenated, 0)

    return embdd_tensor






print("------------------------ Finished ---------------------------------------")