from cleaning_funcs import merge_in
import pandas as pd
import numpy as np

raw_data_file = "../Data/raw_data/conclusions_reembedded.csv"
raw_data = pd.read_csv(raw_data_file)
raw_data = raw_data.iloc[:, 2:]
# sanity check, print keys:
# for key in raw_data.keys():     # comment this out if not needed
#    print(key)                  # comment out
# check for # obs
print("nrow for raw obs number: ")
print(len(raw_data['ArticleID']))

# drop duplicates
final_data = raw_data.drop_duplicates(subset=['ArticleID'])

# merge publicationYear in
merge_file = "../Data/raw_data/merge_data/combined_dataset_full_output.csv"
final_data = merge_in(final_data, file=merge_file, cols=['ArticleID', 'publicationYear'])
final_data['publicationYear'] = [int(x) for x in final_data['publicationYear']]

# merge # of original words in
merge_file = "../Data/raw_data/bow/Kyle/conclusion_early_bow.csv"
final_data = merge_in(final_data, file=merge_file, cols=['ArticleID', 'word_number'])
final_data['word_number'] = [int(x) for x in final_data['word_number']]

# merge # of published words in
bow_data = pd.read_csv("../Data/raw_data/bow/Kyle/conclusion_published_bow.csv", usecols=['ArticleID', 'word_number'])
bow_data = bow_data.rename(columns={'word_number': 'word_number_pub'})
# drop duplicates
bow_data = bow_data.drop_duplicates(subset=['ArticleID'])
# left merge
final_data = final_data.merge(bow_data, how='left', on='ArticleID')
final_data['word_number_pub'] = [int(x) for x in final_data['word_number_pub']]

# merge hedges in
merge_file = "../Data/raw_data/hedge/kyle.csv"
final_data = merge_in(final_data, file=merge_file, cols=['ArticleID',
                                                         'hedge_conclusion_early',
                                                         'hedge_conclusion_published'])
# put in hedge change
final_data['hedge_conclusion_change'] = list(np.subtract(np.array(final_data['hedge_conclusion_published']),
                                            np.array(final_data['hedge_conclusion_early'])))

# merge dictionary in
merge_file = "../Data/raw_data/bow/Kyle/cleaned/conclusion.csv"
final_data = merge_in(final_data, file=merge_file, cols=['ArticleID', 'total_dict_change'])

# calculate hedge and dict change
final_data['hedge_change_with_dict'] = final_data['hedge_conclusion_change'] + final_data['total_dict_change']

# filter out anything that doesn't have a conclusion section (original or published)
final_data = final_data[(final_data['word_number'] > 0) & (final_data['word_number_pub'] > 0)]

print("nrow for final obs number: ")
print(len(final_data['ArticleID']))
final_data.to_csv("../Data/train_test_data/conclusions_kyle.csv")
