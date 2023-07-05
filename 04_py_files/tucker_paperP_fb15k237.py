#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_fb15k237/03_word2vec_fb15k237_300dim_ent_sorted.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_fb15k237/03_word2vec_fb15k237_300dim_rel_sorted.pt', map_location = torch.device(device))

entity_embedd = entity_embedd.contiguous()
relation_embedd = relation_embedd.contiguous()


# Train KGE model with input data → Save results
result = pipeline(
    dataset="fb15k237",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="tucker",
    model_kwargs=dict(
        embedding_dim=entity_embedd.shape[-1],
        relation_dim=relation_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
        dropout_0=0.3,
        dropout_1=0.4,
        dropout_2=0.5,
    ),
    stopper="early",
    stopper_kwargs=dict(frequency=50, patience=5, relative_delta=0.002),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='tuckER',
    ),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.0005, weight_decay=0.0),
    lr_scheduler='ExponentialLR',
    lr_scheduler_kwargs=dict(
            gamma=0.995,
            #step_size=1,
        ),
    loss='bcewithlogits',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=1000, batch_size=128, label_smoothing=0.1,
                         checkpoint_name='tucker_fb15k237_word2vec300_paperP_1000epochs.pt',
                         checkpoint_directory='01_models/tuckER/checkpoints',
                         checkpoint_frequency=30,
                         ),
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
)

result.save_to_directory("01_models/tuckER/fb15k237_word2vec300_paperP_1000epochs.pt")
