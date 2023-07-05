#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/03_word2vec_wn18rr_128dim_ent_sorted.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/03_word2vec_wn18rr_128dim_rel_sorted.pt', map_location = torch.device(device))

# Train KGE model with input data → Save results
result = pipeline(
    dataset="wn18rr",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="conve",
    model_kwargs=dict(
        embedding_dim=entity_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
        feature_map_dropout=0.21969167540833145,
        input_dropout=0.3738110367324488,
        kernel_height=3,
        kernel_width=3,
        output_channels=27,
        output_dropout=0.4598078311847786
    ),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='convE',
    ),
    stopper="early",
    stopper_kwargs=dict(frequency=30, patience=5, relative_delta=0.002),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.0015640153246253687, weight_decay=0.0),
    loss='crossentropy',
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=1000, batch_size=256, label_smoothing=0.0015640153246253687),
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
)

result.save_to_directory("01_models/convE/wn18rr_word2vec_benchP_1000epochs.pt")