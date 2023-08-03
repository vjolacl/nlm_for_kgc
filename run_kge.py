import torch
from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer
import yaml


# Load the configuration file
with open('config_kge.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

device = torch.device("cpu")

# Get the desired model configuration
common_config = config['common']
model = config['model_name']
model_config = config['models'][model]
path = config['path']

# Load pre-generated word embeddings
entity_embedd = torch.load(path['ent_path'], map_location=device)
relation_embedd = torch.load(path['rel_path'], map_location=device)

entity_embedd = entity_embedd.contiguous()
relation_embedd = relation_embedd.contiguous()

# Add pre-trained embeddings to the model parameters
model_config['entity_initializer'] = PretrainedInitializer(tensor=entity_embedd)
model_config['relation_initializer'] = PretrainedInitializer(tensor=relation_embedd)

if model == 'tucker':
    model_config['embedding_dim'] = entity_embedd.shape[-1]
    model_config['relation_dim'] = relation_embedd.shape[-1]
else:
    model_config['embedding_dim'] = entity_embedd.shape[-1] 

result = pipeline(
    model=model,
    model_kwargs=model_config,
    stopper='early',
    stopper_kwargs=dict(frequency=50, patience=3, relative_delta=0.002),
    result_tracker='wandb',
    result_tracker_kwargs=dict(project=str(model)),
    optimizer='adam',
    evaluator="rankbased",
    evaluator_kwargs=dict(filtered=True),
    **common_config,
)

# Save the result
result.save_to_directory(f"01_models/{model}/{path['save_model_name']}")





