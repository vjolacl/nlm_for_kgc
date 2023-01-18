from pykeen.pipeline import pipeline
from pykeen.datasets import Nations
from pykeen.models import TransE
import torch
from typing import List
import pykeen.nn
from pykeen.models.predict import get_tail_prediction_df


dataset = Nations

# Store Nations dataset
nations = Nations()
nations_train = nations.training
nations_test = nations.testing

# Store the entity-to-id and relation-to-id relationship in separate dictionaries
entities_to_ids = nations.entity_to_id
relations_to_ids = nations.relation_to_id

# Testing the KGE pipeline for TransE

results = pipeline(
    dataset='Nations',
    model='TransE',
    training_loop='sLCWA',
)

# Get an idea of a KGE model's results
mdl_results = results._get_results()
print(mdl_results.keys())

# Obtain representation from KGE model
model = results.model # ----> TransE has only one representation for each entity and one for each relation
entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
relation_representation_modules: List['pykeen.nn.Representation'] = model.relation_representations

# Access all entity embeddings, for TranE length of list is 1
entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

# Representations are subclasses of torch.nn.Module, so call them like functions to invoke the forward() method and get the values
entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
relation_embedding_tensor: torch.FloatTensor = relation_embeddings()

# Evaluate KGE model on link prediction

# Try-out tail prediction according to rank-based scoring
df = get_tail_prediction_df(
    model = model,
    head_label = "brazil",
    relation_label = "accusation",
    triples_factory = results.training,
    add_novelties=False,
)
print(df)




print("finished")

