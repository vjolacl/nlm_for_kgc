from pykeen.pipeline import pipeline
from pykeen.datasets import Nations
from functions import *
from pykeen.datasets import WN18RR

dataset = WN18RR()

entities_to_ids, relations_to_ids = get_data(dataset)

# Map entity ID to entity text
df_entity2text = pd.read_csv('data/wn18rr/wn18rr_entity2text.txt', delimiter="\t", header = None, names=["id", "definition"])
df_entity2text[["entity", "description"]] = df_entity2text["definition"].str.split(',', n=1, expand=True)
df_entity2text[:5]
df_entity2text["segmented entities"] = df_entity2text["entity"].str.split(' ')

#b = [[item] for item in a]
print("checkpoint")





# Store KG entities in dataframe to keep track
#df_entities = pd.DataFrame(input_dict.items(), columns=["entity", "index"])

# Store each entity in separate list, then store all lists into a list → needed for word2vec input
row_list = []
for rows in df_entity2text.itertuples():
    my_list = [rows.entity] #Create list for the current row
    row_list.append(my_list) #append the list to the final list
input_word2vec = row_list

# Word2Vec model
w2v_cbow = gensim.models.Word2Vec(input_word2vec, min_count=1, vector_size=100, window=1, sg=0)
word_vectors = w2v_cbow.wv.vectors  # Retrieve word vectors of type numpy array
wv_keys = list(w2v_cbow.wv.index_to_key)  # Retrieve keys to word vectors
wv_dict = res = {wv_keys[i]: word_vectors[i] for i in range(len(wv_keys))}  # Save word vectors with respective key in dictionary

#mapping = wv_dict
#embeddings = torch.from_numpy(word_vectors) #Convert to tensor to use as input for KGE


# Compute average word embeddings for each KG relation
averaged_embeddings = []
for row in df_entity2text["entity"]:
    lst = []
    for item in row:
        lst.append(wv_dict[item])
    avg_embdd = np.mean(lst, axis=0)
    averaged_embeddings.append(avg_embdd)

df_relations["averaged embeddings"] = averaged_embeddings
embeddings = torch.tensor(np.array(averaged_embeddings))  # Convert to tensor to use as input for KGE
mapping = df_relations

print("------------------------ Finished ---------------------------------------")
