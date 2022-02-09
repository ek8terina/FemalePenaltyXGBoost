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

# merge publicationYear in
merge_file = "./Data/raw_data/merge_data/combined_dataset_full_output.csv"
final_data = merge_in(final_data, file=merge_file, cols=['ArticleID', 'publicationYear'])
final_data['publicationYear'] = [int(x) for x in final_data['publicationYear']]

# merge # of original words in
merge_file = "./Data/raw_data/bow/Kyle/abstract_early_bow.csv"
final_data = merge_in(final_data, file=merge_file, cols=['ArticleID', 'word_number'])
final_data['word_number'] = [int(x) for x in final_data['word_number']]

# merge # of published words in
bow_data = pd.read_csv("./Data/raw_data/bow/Kyle/abstract_published_bow.csv", usecols=['ArticleID', 'word_number'])
bow_data = bow_data.rename(columns={'word_number': 'word_number_pub'})
# drop duplicates
bow_data = bow_data.drop_duplicates(subset=['ArticleID'])
# left merge
final_data = final_data.merge(bow_data, how='left', on='ArticleID')
final_data['word_number_pub'] = [int(x) for x in final_data['word_number_pub']]

# merge hedges in
merge_file = "./Data/raw_data/hedge/kyle.csv"
final_data = merge_in(final_data, file=merge_file, cols=['ArticleID', 'hedge_abstract_early', 'hedge_abstract_published'])
# put in hedge change
final_data['hedge_abstract_change'] = list(np.subtract(np.array(final_data['hedge_abstract_published']),
                                            np.array(final_data['hedge_abstract_early'])))

# drop any abstract that are >600 words or <59 words
final_data_few_words = final_data[(final_data['word_number'] < 600) & (final_data['word_number_pub'] < 600)]
final_data_filtered = final_data_few_words[(final_data_few_words['word_number'] > 59) &
                                           (final_data_few_words['word_number_pub'] > 59)]
print("nrow for word number filtered data: ")
print(len(final_data_filtered['ArticleID']))

# filter by word number change as well
# drop anything that has a change >150
final_data_filtered['word_number_change'] = final_data_filtered["word_number_pub"] - final_data_filtered["word_number"]
final_data_filtered = final_data_filtered[((final_data_filtered['word_number_change'] < 150) &
                                           (final_data_filtered['word_number_change'] > -150))]
print("nrow for CHANGE in word number filtered data: ")
print(len(final_data_filtered['ArticleID']))
final_data_filtered.to_csv("./Data/train_test_data/abstracts_kyle.csv")
