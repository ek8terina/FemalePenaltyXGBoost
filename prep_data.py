import pandas as pd
import numpy as np
import re


def read_into_embed_matrix(embeddings_str_vec):
    bracket1 = re.compile(r'^\[\s*')
    bracket2 = re.compile(r'\s*\]$')
    embeddings_str_vec = [re.sub(bracket1, "", x) for x in embeddings_str_vec]
    embeddings_str_vec = [re.sub(bracket2, "", x) for x in embeddings_str_vec]
    embeddings_str_vec = [re.sub(re.compile(r'\s+'), ",", x) for x in embeddings_str_vec]

    embeddings_str_vec = [list(x.split(",")) for x in embeddings_str_vec]
    embeddings = []
    for embed in embeddings_str_vec:
        embeddings.append([float(y) for y in embed])
    embedding_matrix = np.matrix(embeddings)
    return embedding_matrix


raw_data_file = "./Data/raw_data/abstracts_with_embeds.csv"
raw_data = pd.read_csv(raw_data_file)
# check for # obs
print(len(raw_data['ArticleID']))

embedding_matrix = read_into_embed_matrix(raw_data['embedding'])
final_data = pd.concat([raw_data[["ArticleID", "AuthorGend"]], pd.DataFrame(embedding_matrix)], axis=1)

final_data.to_csv("./Data/train_test_data/abstracts_kyle.csv")
