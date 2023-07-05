#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer


device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_fb15k237/02_word2vec_fb15k237_ent_sorted.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_fb15k237/02_word2vec_fb15k237_rel_sorted.pt', map_location = torch.device(device))


# Train KGE model with input data → Save results
result = pipeline(
    dataset="fb15k237",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="conve",
    model_kwargs=dict(
        embedding_dim=entity_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
        feature_map_dropout=0.2,
        input_dropout=0.2,
        output_dropout=0.3
    ),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='convE',
    ),
    lr_scheduler='ExponentialLR',
    lr_scheduler_kwargs=dict(gamma=0.995), # ,step_size=1,),
    stopper="early",
    stopper_kwargs=dict(frequency=5, patience=20),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.003, weight_decay=0.0),
    loss='bcewithlogits',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=1000, batch_size=128, label_smoothing=0.1),
    #regularizer="no",
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
)

result.save_to_directory("01_models/convE/fb15k237_word2vec_paperP_1000epochs.pt")