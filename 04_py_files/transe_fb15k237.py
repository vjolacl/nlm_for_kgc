import torch
from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer

device="cuda"

#Load pre-generated word embeddings
entity_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/05_word2vec_wn18rr_300dim_ent_sorted.pt', map_location = torch.device(device))
relation_embedd = torch.load('03_nlm_embeddings/word2vec_wn18rr/05_word2vec_wn18rr_300dim_rel_sorted.pt', map_location = torch.device(device))

entity_embedd = entity_embedd.contiguous()
relation_embedd = relation_embedd.contiguous()

# Train TransE model with input data → Save results
result = pipeline(
    dataset="fb15k237",
    dataset_kwargs=dict(create_inverse_triples=False),
    model="transe",
    model_kwargs=dict(
        scoring_fct_norm=1,
        embedding_dim=entity_embedd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embedd),
        relation_initializer=PretrainedInitializer(tensor=relation_embedd),
    ),
    stopper="early",
    stopper_kwargs=dict(frequency=50, patience=3, relative_delta=0.002),
    result_tracker='wandb',
    result_tracker_kwargs=dict(
        project='transE',
    ),
    optimizer='adam',
    optimizer_kwargs=dict(lr=0.00091246982681624, weight_decay=0.0),
    loss="bcewithlogits",
    training_loop='lcwa',
    training_kwargs=dict(num_epochs=500, batch_size=128, label_smoothing=0.006091616913055568,
                         checkpoint_name='transe_fb15k237_word2vec300_paperP_500epochs.pt',
                         checkpoint_directory='01_models/transE/checkpoints',
                         checkpoint_frequency=30,
                         ),
    regularizer="no",
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
)

result.save_to_directory("01_models/transE/fb15k237_word2vec300_hpoP_500epochs.pt")