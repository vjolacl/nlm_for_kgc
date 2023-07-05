#import pandas as pd
import torch
from pykeen.pipeline import pipeline
#import numpy as np
#from pykeen.datasets import WN18RR, FB15k237
from pykeen.nn.init import PretrainedInitializer

device="cuda"
model="tucker"
dataset="wn18rr"
filename="01_models/tuckER/wn18rr_word2vec_paperP_1000epochs2.pt"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/04_word2vec_wn18rr_200dim_ent_sorted.pt', map_location=torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/04_word2vec_wn18rr_200dim_rel_sorted.pt', map_location=torch.device(device))


def run_kge_model(dataset, model, entity_embedd, relation_embedd, device, filename, loss, lr, tr_loop, num_epochs, label_smoothing, batch_size, gamma, project_name):
    result = pipeline(
        dataset=dataset,
        dataset_kwargs=dict(create_inverse_triples=False),
        model=model,
        model_kwargs=dict(embedding_dim=entity_embedd.shape[-1],
                          #scoring_fct_norm=1,
                          entity_initializer=PretrainedInitializer(tensor=entity_embedd),
                          relation_initializer=PretrainedInitializer(tensor=relation_embedd),
                          dropout_0=0.2,
                          dropout_1=0.2,
                          dropout_2=0.3),
        device=device,
        stopper="early",
        stopper_kwargs=dict(frequency=5, patience=20, relative_delta=0.002),
        loss=loss,
        #regularizer="NoRegularizer",
        optimizer="adam",
        optimizer_kwargs=dict(weight_decay=0.0, lr=lr),
        training_loop=tr_loop,
        training_kwargs=dict(num_epochs=num_epochs, label_smoothing=label_smoothing, batch_size=batch_size),
        lr_scheduler='ExponentialLR',
        lr_scheduler_kwargs=dict(
            gamma=gamma,
            # step_size=1,
        ),
        result_tracker='wandb',
        result_tracker_kwargs=dict(project=project_name),
        evaluator="RankBasedEvaluator",
        evaluator_kwargs=dict(filtered=True),
        #evaluation_kwargs=dict(batch_size=32),
    )
    result.save_to_directory(filename)


run_kge_model(dataset, model, entity_embedd, relation_embedd, device, filename,
              loss="bcewithlogits", lr=0.01, tr_loop="lcwa", num_epochs=1000, label_smoothing=0.1,
              batch_size=128, gamma=1.0, project_name="tuckER")