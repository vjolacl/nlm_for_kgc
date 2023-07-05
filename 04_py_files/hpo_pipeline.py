import torch
from pykeen.nn.init import PretrainedInitializer
from pykeen.hpo import hpo_pipeline

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/bert_fb15k237/avg_4lastlayers/03_bert_avg4LL_pca200_sorted_ent.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/bert_fb15k237/avg_4lastlayers/03_bert_avg4LL_pca200_sorted_rel.pt', map_location = torch.device(device))


entity_embedd = entity_embedd.contiguous()
relation_embedd = relation_embedd.contiguous()


hpo_pipeline_result = hpo_pipeline(
    dataset='fb15k237',
    dataset_kwargs=dict(create_inverse_triples=False),
    model='transe',
    model_kwargs=dict(embedding_dim=entity_embedd.shape[-1],
                      entity_initializer=PretrainedInitializer(tensor=entity_embedd),
                      relation_initializer=PretrainedInitializer(tensor=relation_embedd)
                     ),
    #model_kwargs_ranges=dict(scoring_fct_norm=dict(type=int, low=1, high=2)),
    device=device,
    negative_sampler_kwargs_ranges=dict(num_negs_per_pos=dict(type=int, low=1, high=256, step=32)),
    stopper="early",
    stopper_kwargs=dict(frequency=50, patience=5, relative_delta=0.002),
    loss="crossentropy",
    optimizer="adam",
    optimizer_kwargs=dict(weight_decay=0.0),
    optimizer_kwargs_ranges=dict(lr=dict(type=float, low=0.0001, high=0.01, scale="log")),
    training_loop="lcwa",
    training_kwargs=dict(num_epochs=100),
    training_kwargs_ranges=dict(label_smoothing=dict(type=float, low=0.000005, high=0.01, scale="log"),
                                batch_size=dict(type=int, low=6, high=9,scale="power_two")),
    result_tracker='wandb',
    result_tracker_kwargs=dict(project='hpo_transe_fb15k237'),
    evaluator="RankBasedEvaluator",
    evaluator_kwargs=dict(filtered=True),
    n_trials=10,
    timeout=90000,
    metric="hits@3",
    direction="maximize",
)

hpo_pipeline_result.save_to_directory("hpo_pipeline/transE/fb15k237_bertavg4LL_pca200_100epochs_24h.pt")