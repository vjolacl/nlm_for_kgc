#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/02_word2vec_wn18rr_64dim_ent_sorted.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/02_word2vec_wn18rr_64dim_rel_sorted.pt', map_location = torch.device(device))


# Train KGE model with input data → Save results
result = pipeline(
    dataset="wn18rr",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="conve",
    model_kwargs=dict(
        embedding_dim=entity_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
        feature_map_dropout=0.005210026247180255,
        input_dropout=0.37519041463524166,
        kernel_height=3,
        kernel_width=3,
        output_channels=31,
        output_dropout=0.08538402811539847
    ),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='convE',
    ),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.0013029734208517471, weight_decay=0.0),
    loss='marginranking',
    loss_kwargs=dict(margin=7.690902408540735),
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=300, batch_size=256),
    #regularizer="no",
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
    #negative_sampler= "basic",
    #negative_sampler_kwargs=dict(num_negs_per_pos=17)
)
result.save_to_directory("01_models/convE/wn18rr_word2vec_benchP_64dim.pt")