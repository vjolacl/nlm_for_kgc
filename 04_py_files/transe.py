#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/01_word2vec_wn18rr_ent_sorted.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/01_word2vec_wn18rr_rel_sorted.pt', map_location = torch.device(device))


# Train TransE model with input data → Save results
result = pipeline(
    dataset="wn18rr",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="transe",
    model_kwargs=dict(
        scoring_fct_norm=1,
        embedding_dim=256 #entity_embedd.shape[-1],
        #entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        #relation_initializer=PretrainedInitializer(tensor=relation_embedd)
    ),
    stopper="early",
    stopper_kwargs=dict(frequency=20, patience=100),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='transE',
    ),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.0011049153751436596, weight_decay=0.0),
    loss='softplus',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=501, batch_size=512, label_smoothing=0.00200051768009458),
    regularizer="no",
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
    #evaluation_kwargs=dict(batch_size=16)
)

result.save_to_directory("01_models/transE/wn18rr_word2vec_paperP_random.pt")