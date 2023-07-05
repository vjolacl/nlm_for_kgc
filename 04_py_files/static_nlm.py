from functions import *
import gensim
from pykeen.datasets import FB15k237
from nltk.tokenize import word_tokenize

device="cpu"
data = FB15k237()


# Store the entity-to-id and relation-to-id relationship in separate dictionaries
ent_to_id = data.entity_to_id
rel_to_id = data.relation_to_id

#Load relation definition
df_rel2text = pd.read_csv('data/fb15k237/fb15k237_relation2text.txt', delimiter="\t", header = None, names=["id", "relation"])
df_rel2text[["property_1_id", "property_2_id"]] = df_rel2text["id"].str.split('.', n=1, expand=True)

#Dataset preprocessing steps
df_rel2text["property_1_id"] = df_rel2text["property_1_id"].str.replace("/", ", ").str[2:]
df_rel2text["property_2_id"] = df_rel2text["property_2_id"].str.replace("/", ", ").str[2:]
df_rel2text["property_1_id"] = df_rel2text["property_1_id"].str.replace("_", " ")
df_rel2text["property_2_id"] = df_rel2text["property_2_id"].str.replace("_", " ")
df_rel2text["segmented"] = df_rel2text["property_2_id"].str.split(',')
df_rel2text.loc[df_rel2text["segmented"].isna(), "segmented"] = df_rel2text.loc[df_rel2text["segmented"].isna(), "property_1_id"].str.split(',')



def test(lst):
    return lst[-2:]

df_rel2text['property_reduced'] = df_rel2text['segmented'].apply(test)
df_rel2text['property_joined'] = [df_rel2text['property_reduced'][i][0] + df_rel2text['property_reduced'][i][1] for i in range(len(df_rel2text))]
df_rel2text['property_tokenized'] = [word_tokenize(df_rel2text['property_joined'][i]) for i in range(len(df_rel2text))]

# Store each relation in separate list, then store all lists into a list → needed for word2vec input
input_word2vec = [df_rel2text["property_tokenized"][i] for i in range(len(df_rel2text))]

# Word2Vec model
w2v_cbow = gensim.models.Word2Vec(input_word2vec, min_count=1, vector_size=256, window=5, sg=1)
word_vectors = w2v_cbow.wv.vectors  # Retrieve word vectors of type numpy array
wv_keys = list(w2v_cbow.wv.index_to_key)  # Retrieve keys to word vectors
wv_dict = res = {wv_keys[i]: word_vectors[i] for i in range(len(wv_keys))}  # Save word vectors with respective key in dictionary


# Compute average word embeddings for each KG relation
averaged_embeddings = []
for idx in range(len(df_rel2text)):
    entity = df_rel2text["id"].loc[idx]
    seg_entity = df_rel2text["property_tokenized"].loc[idx]
    lst = []
    if entity in wv_dict:
        print("if", idx, entity)
        averaged_embeddings.append(wv_dict[entity])
    else:
        print("else", idx, entity)
        for item in seg_entity:
            lst.append(wv_dict[item])
        avg_embdd = np.mean(lst, axis=0)
        averaged_embeddings.append(avg_embdd)

embeddings = torch.tensor(np.array(averaged_embeddings))  # Convert to tensor to use as input for KGE

#torch.save(embeddings, "03_nlm_embeddings/word2vec_fb15k237/01_word2vec_fb15k237_rel.pt")


print("finishes")

