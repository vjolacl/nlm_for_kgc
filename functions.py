
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

def get_data(dataset):
    dataset_train = dataset.training
    dataset_test = dataset.testing

    # Store the entity-to-id and relation-to-id relationship in separate dictionaries
    entities_to_ids = dataset.entity_to_id
    relations_to_ids = dataset.relation_to_id

    return entities_to_ids, relations_to_ids

#---------- BERT embeddings ----------

def generate_bert_embeddings(input_dict):

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # BERT input: get KG entities/relations into separate lists, preserving the sequence in which they are extracted
    df = pd.DataFrame(input_dict.items(), columns= ["entity/relation", "index"])
    kg_input = list(input_dict.keys())

    # Tokenize the KG entities
    input_to_tokens = tokenizer(kg_input, padding=True, return_tensors='pt')['input_ids']

    # Map the token ID to entity
    df["token ID"] = input_to_tokens.tolist()
    
    # Load the BERT model and tokenizer
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval() #Question nr.1: What is the difference in putting model in evaluation mode or not?

    # Generate embeddings
    output = model(input_to_tokens)
    hidden_states = output["hidden_states"]

    return hidden_states, df

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

#--------- KGE word embeddings --------

def retrieve_kge_embeddings(model):

    # Obtain representation from KGE model
    #model = results.model # ----> TransE has only one representation for each entity and one for each relation
    entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
    relation_representation_modules: List['pykeen.nn.Representation'] = model.relation_representations

    # Access all entity embeddings, for TranE length of list is 1
    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

    # Representations are subclasses of torch.nn.Module, so call them like functions to invoke the forward() method and get the values
    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings()

    return entity_embedding_tensor, relation_embedding_tensor



def evaluate_kge_model(kge_model, dataset):

    # Initialize evaluator method
    evaluator = RankBasedEvaluator()

    # Get triples to test (why mapped triples and not only testing??)
    mapped_triples = dataset.testing.mapped_triples #shape: (n, 3)-A 3 column matrix, each row are the head, relation and tail identifier.

    # Evaluate
    eval_results = evaluator.evaluate(
        model=kge_model, 
        mapped_triples=mapped_triples,
        batch_size=1024, #Qestion: What does the batch size influence and how?
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],)

    df_eval_results_raw = eval_results.to_df()

    hitsat10 = eval_results.get_metric("hits_at_10")
    mr = eval_results.get_metric("mean_rank")
    mrr = eval_results.get_metric("mean_reciprocal_rank")
    key_metrics = {"hits_at_10": hitsat10, "mean_rank": mr, "mean_reciprocal_rank": mrr}


    return df_eval_results_raw, key_metrics




print("------------------------ Finished ---------------------------------------")