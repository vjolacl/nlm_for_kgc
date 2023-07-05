#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/bert_wn18rr/avg_4lastlayers/02_bert_avg4lastlayers_pca_sorted_ent.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/bert_wn18rr/avg_4lastlayers/02_bert_avg4lastlayers_pca_sorted_rel.pt', map_location = torch.device(device))

entity_embedd = entity_embedd.contiguous()
relation_embedd = relation_embedd.contiguous()

# Train KGE model with input data → Save results
result = pipeline(
    dataset="wn18rr",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="tucker",
    model_kwargs=dict(
        embedding_dim=entity_embedd.shape[-1],
        relation_dim=relation_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
        apply_batch_normalization=True,
        dropout_0=0.2,
        dropout_1=0.2,
        dropout_2=0.3,
    ),
    stopper="early",
    stopper_kwargs=dict(frequency=50, patience=3, relative_delta=0.002),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='tuckER',
    ),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.01, weight_decay=0.0),
    lr_scheduler='ExponentialLR',
    lr_scheduler_kwargs=dict(gamma=1.0), # step_size=1),
    loss='bcewithlogits',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=1500, batch_size=128, label_smoothing=0.1,
                         checkpoint_name='tucker_wn18rr_bertpca200_paperP_1500epochs.pt',
                         checkpoint_directory='01_models/checkpoints',
                         checkpoint_frequency=30,
                         ),
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
)

result.save_to_directory("01_models/tuckER/wn18rr_word2vec300_paperP_1500epochs.pt")
