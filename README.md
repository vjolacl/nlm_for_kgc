# Master Thesis: Leveraging NLMs for KGC

This repository provides the code developed in the scope of the master thesis "Leveraging Neural Language Models for Knowledge Graph Completion". 
The datasets FB15k-237 and WN18RR are used to evaluate two approaches of employing NLMs for KGC:
1. NLM-enahenced KGE Models  (TransE, ConvE, TuckER, RotatE)
2. KG-NLM (RoBERTa, DistilBERT, BLOOM)

Link to all visualizations of models' performance:  https://wandb.ai/nlm_kgc?shareProfileType=copy
A description of the approaches and the experimental results are provided in the .pdf file. Tables with the selected hyperparameters can be found in the appendix in the .pdf file.
## Repository structure

#### - 00_data: 
provides entity and relation textual descriptions for both datasets 

#### - 01_kg_nlm_datasets: 
provides precomputed train/test/val sets for both datasets  

#### - 02_nlm_embeddings: provides precomputed NLM-generated (Word2Vec and BERT) embeddings used in the first approach.
#### - bert_embeddings.ipynb: jupyter notebook used to generate BERT embeddings
#### - word_embeddings.ipynb: jupyter notebook used to generate Word2Vec embeddings
#### - config_kg_nlm.yaml: config file to provide all input data and hyperparameters to train the KG-NLM models
#### - config_kge.yaml: config file to provide all input data and hyperparameters to train the KGE models with NLM embeddings
#### - data_preprocessing_kg_nlm.ipynb: jupyter notebook used to generate input data for the KG-NLM approach
#### - dim_reduction.ipynb: jupyter notebook to generate the PCA NLM embeddings
#### - functions.py: python file containing all relevant functions 
#### - job.sh: bash script file to run models on server 
#### - rotate.ipynb: jupyter notebook tu run rotatE model
#### - run_kg_nlm.py: script to run KG-NLM models 
#### - run_kge.py: script to run KGE models 



    