
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
nltk.download('punkt')


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

#---------- Word2Vec embeddings ----------


# Generate word embeddings from Word2Vec

def generate_word2vec_embeddings(input_dict, entity):
    # Download otherwise package does not function

    if entity=True:
        # Store KG entities in dataframe to keep track
        df_entities = pd.DataFrame(input_dict.items(), columns=["entity", "index"])

        # Store each entity in separate list, then store all lists into a list → needed for word2vec input
        row_list = []
        for rows in df_entities.itertuples():
            my_list = [rows.entity] #Create list for the current row
            row_list.append(my_list) #append the list to the final list
        input_word2vec = row_list

        # Word2Vec model
        w2v_cbow = gensim.models.Word2Vec(input_word2vec, min_count=1, vector_size=4, window=1, sg=0)
        word_vectors = w2v_cbow.wv.vectors  # Retrieve word vectors
        wv_keys = list(w2v_cbow.wv.index_to_key)  # Retrieve keys to word vectors
        wv_dict = res = {wv_keys[i]: word_vectors[i] for i in range(len(wv_keys))}  # Save word vectors with respective key in dictionary

        mapping = wv_dict
        embeddings = word_vectors
    else:

        # Store KG relations in dataframe to keep track
        df_relations = pd.DataFrame(input_dict.items(), columns= ["relation", "index"])

        # Separate words within KG relations
        load()
        df_relations["segmented relations"] = [segment(relation) for relation in df_relations["relation"]]
        df_relations_expanded = df_relations[["index", "segmented relations"]].explode("segmented relations")
        df_relations_expanded = df_relations_expanded.reset_index(drop=True)  #reset index

        # Transform segmented relations to list -> ready to feed to Word2Vec model
        #input_word2vec = df_relations_expanded["segmented relations"].tolist()
        input_word2vec = [df_relations["segmented relations"][i] for i in range(len(df_relations["segmented relations"]))] # treat combined relations as sentences

        # Word2Vec model
        w2v_cbow = gensim.models.Word2Vec(input_word2vec, min_count = 1,vector_size = 4, window = 1, sg=0)
        word_vectors = w2v_cbow.wv.vectors        # Retrieve word vectors
        wv_keys = list(w2v_cbow.wv.index_to_key)  # Retrieve keys to word vectors
        wv_dict = res = {wv_keys[i]: word_vectors[i] for i in range(len(wv_keys))} # Save word vectors with respective key in dictionary

        # Compute average word embeddings for each KG relation
        averaged_embeddings = []
        for row in df_relations["segmented relations"]:
            lst = []
            for item in row:
                lst.append(wv_dict[item])
            avg_embdd = np.mean(lst, axis=0)
            averaged_embeddings.append(avg_embdd)

        df_relations["averaged embeddings"] = averaged_embeddings
        embeddings = averaged_embeddings
        mapping = df_relations


    return mapping, embeddings



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