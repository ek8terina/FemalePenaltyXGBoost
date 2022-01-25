# FemalePenaltyXGBoost
This is a model for Anna Costello's Article Sentiment in economics, finance, and accounting journals project. The goal is to model the change in hedge words between early and published versions of papers from these journals using the gender of their authors and the text (BERT embeddings) as features. 
## Method Overview
...
## Data
### Hedge words
Hedging is a common linguistic practice to indicate uncertainty or cautionary language. Detecting uncertainty cues is the main goal of a breadth of literature. Sentences like:
1. "You *may* leave."
2. "This *may* indicate that."

Differ because although they both contain the work "may" only in sentence (2) is it an uncertainty cue. In the [CoNLL-2010 shared task](https://aclanthology.org/W10-3001.pdf), dataset of annotated uncertainty cues (including type of uncertainty) is created from several different sources. Said dataset has been updated for [Cross-Genre and Cross-Domain Detection of Semantic Uncertainty](https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00098) (Szarvas et al.) and is available [here](https://rgai.sed.hu/file/139). Using this dataset, we use a fine-tuned [SciBERT model](https://github.com/PeterZhizhin/BERTUncertaintyDetection) created by Peter Zhizhin to predict uncertainty cues across a dataset of 5600 articles (each including an early and published version). Change in hedge words is calculated by subtracting the early version's number of predicted hedges from the published version's number of predicted hedges.
### Text + Sections
...
## Training
...
## Results
...
## Prediction
...
