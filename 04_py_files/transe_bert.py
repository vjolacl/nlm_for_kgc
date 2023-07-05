#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/bert_wn18rr/avg_4lastlayers/03_bert_extended_wn18rr_sorted_ent.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/bert_wn18rr/avg_4lastlayers/00_bert_avg4lastlayers_wn18rr_sorted_rel.pt', map_location = torch.device(device))

entity_embedd = entity_embedd.contiguous()
relation_embedd = relation_embedd.contiguous()

# Train TransE model with input data → Save results
result = pipeline(
    dataset="wn18rr",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="transe",
    model_kwargs=dict(
        scoring_fct_norm=1,
        embedding_dim=entity_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd)
    ),
    stopper="early",
    stopper_kwargs=dict(frequency=20, patience=100, relative_delta=0.002),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='transE',
    ),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.0011049153751436596, weight_decay=0.0),
    loss='softplus',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=300, batch_size=512, label_smoothing=0.00200051768009458),
    regularizer="no",
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
    evaluation_kwargs=dict(batch_size=16)
)

result.save_to_directory("01_models/transE/wn18rr_bertavg4LL_extended_300epochs.pt")