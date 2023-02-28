from torch.nn.init import xavier_uniform_
from pykeen.pipeline import pipeline
from pykeen.datasets import Nations
from pykeen.models import TransE
import torch
from typing import List
import pykeen.nn
from pykeen.models import predict
from pykeen.evaluation import RankBasedEvaluator
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
from wordsegment import load, segment
from pykeen.models import TransE
from pykeen.nn.init import PretrainedInitializer
from functions import *

# Get input data for KGE model
entities, relations = get_data(dataset=Nations())

# Generate BERT word embeddings for entities and relations separately
#bert_output_ent = generate_bert_embeddings(entities)
#bert_output_rel = generate_bert_embeddings(relations)
#entity_embdd = concat_hidden_states(bert_output_ent[0])
#relation_embdd = concat_hidden_states(bert_output_rel[0])

# Generate Word2Vec embeddings for entities and relations separately
w2v_ent_dict, entity_embdd = generate_word2vec_embeddings(entities, entity=True)
w2v_rel_df, relation_embdd = generate_word2vec_embeddings(relations, entity=False)

# Save word embeddings for both entities and relations including all hidden states
#torch.save(bert_output_ent[0], '04_nlm_word_embeddings/bert_hiddenstates_nations_ent.pt')
#torch.save(bert_output_rel[0], '04_nlm_word_embeddings/bert_hiddenstates_nations_rel.pt')
#torch.save(entity_embdd, '04_nlm_word_embeddings/word2vec_nations_ent.pt')
#torch.save(relation_embdd, '04_nlm_word_embeddings/word2vec_nations_rel.pt')


# Train KGE model with input data → Save results
result = pipeline(
    dataset="nations",
    model="transe",
    model_kwargs=dict(
        embedding_dim=entity_embdd.shape[-1],
        entity_initializer=PretrainedInitializer(tensor=entity_embdd),
        relation_initializer=PretrainedInitializer(tensor=relation_embdd),
    ),
)
result.save_to_directory("01_models/nations_transE_word2vec_no1")


# Retrieve word embeddings from KGE model → Save results
#call function here from function.py file


# Evaluate KGE model on link prediction → Save results
eval_results_raw, key_metrics = evaluate_kge_model(result.model, dataset=Nations())
#eval_results_raw.to_csv('02_evaluation_results/eval_results_nations_transE_word2vec.csv')


print("done")