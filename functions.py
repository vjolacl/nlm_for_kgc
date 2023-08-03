import torch
import os
from transformers import BertTokenizer, BertModel
#import gensim
import requests
import nltk
import pandas as pd
import numpy as np
import torch
from pykeen.sampling.basic_negative_sampler import BasicNegativeSampler
#from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn

class customdata(torch.utils.data.Dataset):

    def __init__(self, token_tensor, attention_tensor):
        self.token_tensor = token_tensor
        self.attention_tensor = attention_tensor

    def __len__(self):
        return self.token_tensor.size()[0]

    def __getitem__(self, idx):
        item = self.token_tensor[idx]
        attention_item = self.attention_tensor[idx]
        output = {"input_id": item, "attention_mask": attention_item}

        return output

def generate_bert_embeddings(data, model, device):
    """
    This function generates BERT embeddings for the provided input
    Args:
        data: provide input tokens
        model: provide already loaded pretrained BERT model
        device: "cpu" or "cuda"

    Returns:
        Return the embeddings for the last four hidden layers of BERT
    """
    #Put model into evaluation mode
    model.eval()

    # Generate embeddings
    output = model(**data)
    hidden_states = output["hidden_states"]
    last4_layers = hidden_states[-4:]
    
    return last4_layers


def load_ent_embeddings(path, device):
    """
    This function loads the BERT entity embedding files (saved in batches)
    Args:
        path: provide path of directory where the raw bert entity embedding files are saved
        device: "cpu" or "cuda"

    Returns:
        List of BERT entity embeddings; needs further processing due to batch structure shape(17,4,2500,19,768)

    """
    files = os.listdir(path)
    files = sorted(files)

    # Ignore first two files in directory: one is relation embedddings and the other one is jupyter nootebook file
    ent = files[2:]
    print(files)
    
    ent_embedd_raw = []
    for file in ent:
        batch_embedd = torch.load(path + '/' + file, map_location = torch.device(device))
        batch_cpu = batch_embedd.to(device)
        ent_embedd_raw.append(batch_cpu)
        del batch_embedd
        torch.cuda.empty_cache()
    
    return ent_embedd_raw


def load_bert(device):
    """
    This function in used to load the pretrained 'bert-base-uncased' tokenizer and model, outputting the hidden states.
    Args:
        device: provide device to load tokenizer and model "cpu" vs. "cuda"

    Returns: pretrained BERT tokenizer and model

    """
    # Load the BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = model.to(device)
    
    return tokenizer, model


def tokenize(tokenizer, input_list):
    """
    This functions tokenizes the given input using the given tokenizer. It applies padding and truncation.
    Args:
        tokenizer: provide desired tokenizer
        input_list: provide list of strings to tokenize

    Returns: dictionary with tokenized input, given as dict["input_ids"] and attention mask as dict["attention_mask"]

    """
    # Tokenize the KG entities
    input_to_tokens = tokenizer(input_list, padding=True, truncation=True, return_tensors='pt')
    
    return input_to_tokens


def batch_embedd(dataloader, model, output_path, filename, device):
    """
    This function is used to generated BERT embeddings of a KG's entities. The input is processed in batches using a dataloader.7
    The entity embeddings, are saved in separate files as they are processed in batches.
    Args:
        dataloader: provide dataloader already instantiated with the tokenized KG entities
        model: provide pretrained model, that has already been loaded
        output_path: provide the path of the directory where the BERT embeddings should be saved
        filename: provide the name of the files in which the embeddings are to be saved
        device: "cpu" or "cuda"

    Returns: none

    """
    iter = 1
    embeddings = []
    for data in dataloader:
        output = generate_bert_embeddings(data, model, device)
        output = torch.stack(output, 0)# output is originally tuple → stack BERT's last four hidden layers tensors into 1 tensor

        # Used to keep count of the files created and give them numbered names
        if iter < 10:
            batch_nr = "00" + str(iter)
        else:
            if iter < 100:
                batch_nr = "0" + str(iter)
            else: 
                batch_nr = str(iter)
        
        torch.save(output, output_path + batch_nr + filename)

        iter = iter + 1
        del data
        del output
        torch.cuda.empty_cache()
        
        
def generate_w2v_embeddings(wv_input, w2v_vocab, w2v_vectors):

    word2vec_embeddings = []
    exclude = []
    for array in wv_input:
        embeddings = []
        for word in array:
            if word not in w2v_vocab:
                exclude.append(word)
            else:
                vw_embedding = w2v_vectors[word]
                embeddings.append(vw_embedding)
        word2vec_embeddings.append(embeddings)
    return word2vec_embeddings, exclude



def triple_labels(dataset):
    """
    This function is used to get the triples in label form from pykeen's triple factory
    Args:
        dataset: provide pykeen dataset

    Returns:
        Triple labels separated into train, test and validation sets
    """
    # Get triples in label form e.g. ['/m/010016', '/location/', '/m/0mr_8']
    train = dataset.training.triples
    test = dataset.testing.triples
    val = dataset.validation.triples

    return train, test, val


def triple_ids(dataset):
    """
    This function is used to get the triples in ID form from pykeen's triple factory
    Args:
        dataset: provide pykeen dataset

    Returns:
        Triple ids separated into train, test and validation sets
    """
    # Get triples in ID form e.g. [0, 120, 13647]
    train = dataset.training.mapped_triples
    test = dataset.testing.mapped_triples
    val = dataset.validation.mapped_triples

    return train, test, val


def neg_sampling(sampler, triple_ids, triple_factory):
    """
    This function computes negative sampling for the provided triples with filtered setting.
    Args:
        sampler: provide BasicNegativeSampler from pykeen
        triple_ids: provide triple ids
        triple_factory: provide pykeens triple factory of the selected dataset, e,g. dataset.training

    Returns:
        The negative sampled triples
    """

    # Initialize negative sampler from pykeen
    neg_sampler = BasicNegativeSampler(mapped_triples=triple_ids, filtered=True)

    # Compute negative samples for the given triples
    neg_triples, filter_mask = neg_sampler.sample(triple_ids)

    # Create mask to filter out the neg_triples that are included in the initial positive triples of the KG
    mask = np.ones(len(neg_triples), dtype=bool)
    mask[np.where(filter_mask == False)[0]] = False
    tensor_filtered = neg_triples[mask]  # Apply the mask to remove duplicates

    # Reshape tensor to match the shape of mapped_triples from the pykeen triple factory
    tensor_filtered = tensor_filtered.reshape(len(tensor_filtered), 3)
    triple_labels = triple_factory.label_triples(tensor_filtered)  # enter triple ID (number) and get the triple labels

    del tensor_filtered

    return triple_labels


def load_ent_rel_def(dataset, path_ent, path_rel):
    """
    This function is used to load the dataset specific files containing entities and relations. It supports only the datasets "fb15k-237" and "wn18rr".
    Args:
        dataset: provide name of dataset
        path_ent: provide path to file of datatset entities
        path_rel: provide path to file of dataset relations

    Returns:
        Two dataframes, corresponding to datasets' entities and relations respectively
    """



    if dataset == "fb15k237":
        df_entity2text = pd.read_csv(path_ent, delimiter="\t", header=None, names=["id", "entity"])
        df_entity2text["segmented_entities"] = df_entity2text["entity"].str.split(' ')

        df_rel2text = pd.read_csv(path_rel, delimiter="\t", header=None, names=["id", "definition"])
        df_rel2text[["property_1_id", "property_2_id"]] = df_rel2text["id"].str.split('.', n=1, expand=True)
        df_rel2text["property_1_id"] = df_rel2text["property_1_id"].str.replace("/", ", ").str[2:]
        df_rel2text["property_2_id"] = df_rel2text["property_2_id"].str.replace("/", ", ").str[2:]
        df_rel2text["property_1_id"] = df_rel2text["property_1_id"].str.replace("_", " ")
        df_rel2text["property_2_id"] = df_rel2text["property_2_id"].str.replace("_", " ")

    elif dataset == "wn18rr":
        df_entity2text = pd.read_csv(path_ent, delimiter="\t", header=None, names=["id", "definition"])
        df_entity2text[["entity", "description"]] = df_entity2text["definition"].str.split(',', n=1, expand=True)
        df_entity2text.id = df_entity2text.id.astype(str)
        df_entity2text["id"] = df_entity2text["id"].str.rjust(8, '0')

        df_rel2text = pd.read_csv(path_rel, delimiter="\t", header=None, names=["id", "definition"])

    else:
        print("Only datasets 'fb1k237' and 'wn18rr' are supported")

    return df_entity2text, df_rel2text


def triple_def(df_entity2text, df_rel2text, triples, target):
    """
    This function creates a dataframe containing a datasets' triple in textual form, along with the respective classification label.
    Args:
        df_entity2text: provide dataframe of entities
        df_rel2text: provide dataframe of relations
        triples: provide triples
        target: indicate if current triples are positive and negative

    Returns:
        Dataframe containing the triples in textual form along with the classification label
    """

    df = pd.DataFrame(triples, columns=['head', 'rel', 'tail'])

    df['head_label'] = df['head'].map(df_entity2text.set_index('id')['entity'])
    df['rel_label'] = df['rel'].map(df_rel2text.set_index('id')['definition'])
    df['tail_label'] = df['tail'].map(df_entity2text.set_index('id')['entity'])

    if target == "pos":
        df["target"] = 1
    elif target == "neg":
        df["target"] = 0
    else:
        print("This parameter can only be either 'pos' or 'neg'")

    if df.isnull().values.any():
        print("Dataframe contains nan values, review input")
    return df


def merge_pos_neg_triples(pos_triples, neg_triples):
    """
    This function is used to conatenate the positive and negative triples for a given dataset.
    Args:
        pos_triples: provide dataframe of positive triples
        neg_triples: provide dataframe of negative triples

    Returns:
        A dictionary containing a list of the textual triples and a list of the respective classification labels
    """
    # Concatenate the positive and negative triples in the dataframe and shuffle the data
    df = pd.concat([pos_triples, neg_triples])
    df = df.reset_index(drop=True)
    df = df.sample(frac=1, random_state=5)


    df["triple"] = df["head_label"] + " " + df["rel_label"] + " " + df["tail_label"]

    # Store triples in a dictionary along with the labels
    output = {"triples": df["triple"].tolist(), "labels": df["target"].tolist()}

    return output


class kg_nlm_nput(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_dict(path):
    f = open(path,'r')
    data=f.read()
    f.close()
    return eval(data)

def save_dict(dic, name):
    f = open(name,'w')
    f.write(str(dic))
    f.close()

                
def rotate_bert_init(re_layer, im_layer):
                
    # Retrieve embeddings for CLS layer and turn to tensor 
    re = torch.stack([re_layer[rel][0] for rel in range(len(re_layer))])
    im = torch.stack([im_layer[rel][0] for rel in range(len(im_layer))])

    re_reshaped = re.unsqueeze(2)
    im_reshaped = im.unsqueeze(2)
          
    combined_tensor = torch.cat((re_reshaped, im_reshaped), dim=2)
    return combined_tensor

def concat_ent_embeddings(ent_embedd_raw):
    """
    This function concatenates the raw entity embeddings to remove the batch structure of the files.
    Args:
        ent_embedd_raw: provide multidimensional list where raw entity embeddings are saved after being loaded

    Returns:
        List of entity embeddings in the shape (4,nr_samples,nr_tokens,bert_dimension)
    """
    # Raw BERT embeddings are saved into batches → Transform dimensions so that there are 4 hidden layers each containing all entity tensors 
    embeddings = []
    for layer in range(len(ent_embedd_raw[0])):
        lst = [ent_embedd_raw[i][layer] for i in range(len(ent_embedd_raw))]
        layer_embdd = torch.cat(lst, dim=0)
        del lst
        embeddings.append(layer_embdd)
        del layer_embdd
        torch.cuda.empty_cache()
    del ent_embedd_raw # Clear memory 
    return embeddings


def init_phases(x):
    if x.shape[-1] != 2:
            new_shape = (*x.shape[:-1], -1, 2)
            x = x.view(*new_shape)
    phases = 2 * np.pi * torch.view_as_complex(x).real
    return torch.view_as_real(torch.complex(real=phases.cos(), imag=phases.sin())).detach()


def avg_bert4LL_embeddings(embeddings):
    """
    This function averages the embeddings of BERT's four last layers for the CLS token.
    Args:
        embeddings: provide as input a list of entity embeddings in shape(4, nr_samples, nr_tokens, bert_dimension)

    Returns:
        Tensor of averaged entity embeddings in shape(nr_samples, bert_dimensions)
    """
    # Average the last 4 hidden layers for token 0 = CLS token, cls_embdd = hidden_states[:][:][0]
    embedd_averaged = []

    # For each text sequence...
    for index in range(len(embeddings[0])):
        # average the vectors from the last four layers for the CLS token. Each layer vector is 768 values, so `avg_vec` is length 768.
        tensor = torch.stack((embeddings[-1][index][0], embeddings[-2][index][0],
                              embeddings[-3][index][0], embeddings[-4][index][0]), dim=0)
        avg_vec = torch.mean(tensor, dim=0)
        embedd_averaged.append(avg_vec)

    # Convert output to 2D tensor
    ent_embedd_avg = torch.stack(embedd_averaged, 0)

    # Clear up memory
    del tensor
    del avg_vec
    del embedd_averaged

    return ent_embedd_avg



def remove_characters(strings):
    """
    This function cleans entity and relation textual form from unwanted characters to make the text suitable for Word2Vec embedding matching
    Args:
        strings: string input sequence to clean

    Returns: cleaned string

    """
    char1 = [',', "'", ':', ';', "!"]
    char2 = ['-', "/"]
    result = []
    for string in strings:
        for char in char1:
            string = string.replace(char, '')
        for char in char2:
            string = string.replace(char, ' ')
        result.append(string)
    return result


def embedding_mapping(df, id_column, id_list, embeddings):
    """
    This function sorts the BERT averaged embeddings according to the sequence of the dataset from pykeen.
    Args:
        df: provide dataframe of entities
        id_column: provide id column within dataframe
        id_list: provide id list of entities
        embeddings: provide averaged embeddings tensor

    Returns:
        Tensor of sorted embeddings
    """
    mapping = {}
    for item in id_list:
        # Get index of item in df entity2text
        idx = df[id_column == item].index.item()

        # Add tensor to dictionary, whereby key is the entity id
        mapping[item] = embeddings[idx]

    to_list = list(mapping.values())
    sorted_embeddings = torch.stack(to_list, 0)

    return sorted_embeddings


def compute_metrics(pred):
    """
    This function computed the metrics related to the output of the KG-NLM
    Args:
        pred: provide here model predictions

    Returns:
        A dictionary containing the metrics: eccuracy, f1-score, precision and recall.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def avg_w2v_embeddings(raw_embeddings):
    """
    This function computes the averaged Word2Vec embeddings.
    Args:
        raw_embeddings: provide list of the Word2Vec embeddings in the the format (#samples, #words, #dimensions)

    Returns:
        Tensor of averaged embeddings of shape (#samples, #dimensions)
    """
    # Compute average word embeddings for each KG entity/relation
    averaged_embeddings = []
    for item in raw_embeddings:
        avg_embedd = np.mean(item, axis=0)
        averaged_embeddings.append(avg_embedd)
    embeddings = torch.tensor(np.array(averaged_embeddings))

    return embeddings
