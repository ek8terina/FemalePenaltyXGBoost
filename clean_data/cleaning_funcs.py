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


def merge_in(data, file, cols, drop_dupes=True):
    merge_data = pd.read_csv(file, usecols=cols)
    if drop_dupes:
        merge_data = merge_data.drop_duplicates(subset=['ArticleID'])
    data = data.merge(merge_data, how='left', on='ArticleID')
    return data
