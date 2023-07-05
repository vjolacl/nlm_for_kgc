import torch
from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/bert_wn18rr/avg_4lastlayers/03_bert_extended_wn18rr_sorted_ent.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/bert_wn18rr/avg_4lastlayers/00_bert_avg4lastlayers_wn18rr_sorted_rel.pt', map_location = torch.device(device))

# Train KGE model with input data → Save results
result = pipeline(
    dataset="wn18rr",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="boxe",
    model_kwargs=dict(
        embedding_dim=entity_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
    ),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='boxE',
    ),
    loss_kwargs=dict(margin=5),
    stopper="early",
    stopper_kwargs=dict(frequency=50, patience=150, relative_delta=0.002),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.001, weight_decay=0.0),
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=500),
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
)

result.save_to_directory("01_models/boxE/wn18rr_bertdescript_avg4LL_paperP_500epochs.pt")