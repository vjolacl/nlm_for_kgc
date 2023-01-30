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

from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from wordsegment import load, segment

#dataset = Nations possibly to be deleted

# Store Nations dataset
nations = Nations()
nations_train = nations.training
nations_test = nations.testing

# Store the entity-to-id and relation-to-id relationship in separate dictionaries
entities_to_ids = nations.entity_to_id
relations_to_ids = nations.relation_to_id

# Testing the KGE pipeline for TransE

# Train TransE model with Nations dataset
#results = pipeline(dataset='Nations', model='TransE', training_loop='sLCWA')

#Save model in models directory
#results.save_to_directory("models/nations_transE")

# Load pre-trained model
model = torch.load("models/nations_transE/trained_model.pkl")


# Get an idea of a KGE model's results
#mdl_results = results._get_results()
#print(mdl_results.keys())

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

# Evaluate KGE model on link prediction

# Try-out tail prediction for one example, according to rank-based scoring
df = predict.get_prediction_df(
    model = model,
    head_label = "brazil",
    relation_label = "accusation",
    triples_factory = nations.training,
    add_novelties=False,
)
print(df)

# Get scores for all possible triples
predictions_df = predict.get_all_prediction_df(model, triples_factory=nations.training)
predictions_df

# Get scores for 20 highest scoring triples
predictions_df_20 = predict.get_all_prediction_df(model, k=20, triples_factory=nations.training)
predictions_df_20


# Evaluate the trained TransE model

# Initialize evaluator method
evaluator = RankBasedEvaluator()

# Get triples to test (why mapped triples and not only testing??)
mapped_triples = nations_test.mapped_triples

# Evaluate
eval_results = evaluator.evaluate(
    model=model,
    mapped_triples=mapped_triples,
    batch_size=1024,
    additional_filter_triples=[
        nations.training.mapped_triples,
        nations.validation.mapped_triples,
    ],
)
df_eval_results = eval_results.to_df()


print("---------------------------------------------------------------")



# TEST: Extract word embeddings from BERT using Nations dataset)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# BERT input: get KG entities into list, preserving the sequence in which they are extracted
df_entities = pd.DataFrame(entities_to_ids.items(), columns= ["entity", "index"])
kg_entities = list(entities_to_ids.keys())

# Tokenize the KG entities
entities_to_tokens = tokenizer.batch_encode_plus(kg_entities, return_tensors='pt')['input_ids']

# Map the token ID to entity
df_entities["token ID"] = entities_to_tokens.tolist()

# Load the BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval() # Question nr.1: What is the difference in putting model in evaluation mode or not?

# Understanding the output
output = model(entities_to_tokens)
print("Pre-trained BERT output consists of:", output.keys())
hidden_states = output["hidden_states"]

print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0
print ("Number of text sequences:", len(hidden_states[layer_i]))
batch_i = 0
print ("Number of tokens (sequence length):", len(hidden_states[layer_i][batch_i]))
token_i = 0
print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

# Concat last 4 hidden states for token 0 = CLS token
#cls_embdd = hidden_states[:][:][0]
embdd_concatenated = []

# For each text sequence...
for index in range(len(hidden_states[0])):
    # Concatenate the vectors from the last four layers. Each layer vector is 768 values, so `cat_vec` is length 4*768.
    cat_vec = torch.cat((hidden_states[-1][index][0], hidden_states[-2][index][0],
                         hidden_states[-3][index][0], hidden_states[-4][index][0]), dim=0)
    embdd_concatenated.append(cat_vec)


print("---------------------------------------------------------------")


# Generate word embeddings from Word2Vec

# Store KG relations in dataframe to keep track
df_relations = pd.DataFrame(relations_to_ids.items(), columns= ["relation", "index"])

# Download otherwise package does not function
nltk.download('punkt')

# Separate words within KG relations
load()
df_relations["segmented relations"] = [segment(relation) for relation in df_relations["relation"]]












print("---------------------------------------------------------------")
