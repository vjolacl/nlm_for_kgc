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

# Get input data for KGE model
#entity_embdd
#relation_embdd


# Train KGE model with input data → Save results
result = pipeline(
    dataset="nations",
    model="transe",
    model_kwargs=dict(
        embedding_dim = entity_embdd.shape[-1],
        entity_initializer = PretrainedInitializer(tensor=entity_embdd),
        relation_initializer = PretrainedInitializer(tensor=relation_embdd),
    ),
)
#results.save_to_directory("models/nations_transE_no1")



# Retrieve word embeddings from KGE model → Save results



# Evaluate KGE model on link prediction → Save results




print("done")