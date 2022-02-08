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


# read in embeds
raw_data_file = "./Data/raw_data/abstracts_with_embeds.csv"
raw_data = pd.read_csv(raw_data_file)
# check for # obs
print(len(raw_data['ArticleID']))
# change embeddings to be separate variables:
embedding_matrix = read_into_embed_matrix(raw_data['embedding'])
final_data = pd.concat([raw_data[["ArticleID", "AuthorGend"]], pd.DataFrame(embedding_matrix)], axis=1)
# drop duplicates
final_data = final_data.drop_duplicates(subset=['ArticleID'])

# merge additional vars in
merge_data = pd.read_csv("./Data/raw_data/merge_data/combined_dataset_full_output.csv",
                         usecols=['ArticleID', 'publicationYear'])
# drop duplicates
merge_data = merge_data.drop_duplicates(subset=['ArticleID'])
# left merge
final_data = final_data.merge(merge_data, how='left', on='ArticleID')
final_data['publicationYear'] = [int(x) for x in final_data['publicationYear']]

# merge # of words in
bow_data = pd.read_csv("./Data/raw_data/bow/Kyle/abstract_early_bow.csv",
                         usecols=['ArticleID', 'word_number'])
# drop duplicates
bow_data = bow_data.drop_duplicates(subset=['ArticleID'])
# left merge
final_data = final_data.merge(bow_data, how='left', on='ArticleID')
final_data['word_number'] = [int(x) for x in final_data['word_number']]

# merge hedges in
kyle_hedges = pd.read_csv("./Data/raw_data/hedge/kyle.csv",
                          usecols=['ArticleID', 'hedge_abstract_early', 'hedge_abstract_published'])
# drop duplicates
kyle_hedges = kyle_hedges.drop_duplicates(subset=['ArticleID'])
# left merge
final_data = final_data.merge(kyle_hedges, how='left', on='ArticleID')
# put in hedge change
final_data['hedge_abstract_change'] = list(np.subtract(np.array(final_data['hedge_abstract_published']),
                                            np.array(final_data['hedge_abstract_early'])))
# save to final
final_data.to_csv("./Data/train_test_data/abstracts_kyle.csv")
