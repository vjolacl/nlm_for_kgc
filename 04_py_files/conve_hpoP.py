#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer


device="cuda"

#Load pre-generated word embeddings
#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/05_word2vec_wn18rr_300dim_ent_sorted.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/05_word2vec_wn18rr_300dim_rel_sorted.pt', map_location = torch.device(device))

entity_embedd = entity_embedd.contiguous()
relation_embedd = relation_embedd.contiguous()

# Train KGE model with input data → Save results
result = pipeline(
    dataset="wn18rr",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="conve",
    model_kwargs=dict(
        embedding_dim=entity_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
        feature_map_dropout=0.5,
        input_dropout=0.4,
        kernel_height=3,
        kernel_width=3,
        output_channels=16,
        output_dropout=0
    ),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='convE',
    ),
    stopper="early",
    stopper_kwargs=dict(frequency=50, patience=3, relative_delta=0.002),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.009330514459787512, weight_decay=0.0),
    loss='bcewithlogits',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=1000, batch_size=128, label_smoothing=0.004869831870660787,
                         checkpoint_name='convE_wn18rr_word2vec300_hpoP_500epochs.pt',
                         checkpoint_directory='01_models/convE/checkpoints',
                         checkpoint_frequency=30,
                         ),
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
)

result.save_to_directory("01_models/convE/wn18rr_word2vec300_hpoP_1000epochs.pt")