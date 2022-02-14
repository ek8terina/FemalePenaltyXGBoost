# FemalePenaltyXGBoost
This is a model for Anna Costello's Article Sentiment in economics, finance, and accounting journals project. The goal is to model the change in hedge words between early and published versions of papers from these journals using the gender of their authors and the text (BERT embeddings) as features. 
## Data
### Hedge words
Hedging is a common linguistic practice to indicate uncertainty or cautionary language. Detecting uncertainty cues is the main goal of a breadth of literature. Sentences like:
1. "You *may* leave."
2. "This *may* indicate that."

Differ because although they both contain the work "may" only in sentence (2) is it an uncertainty cue. In the [CoNLL-2010 shared task](https://aclanthology.org/W10-3001.pdf), dataset of annotated uncertainty cues (including type of uncertainty) is created from several different sources. Said dataset has been updated for [Cross-Genre and Cross-Domain Detection of Semantic Uncertainty](https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00098) (Szarvas et al.) and is available [here](https://rgai.sed.hu/file/139). Using this dataset, we use a fine-tuned [SciBERT model](https://github.com/PeterZhizhin/BERTUncertaintyDetection) created by Peter Zhizhin to predict uncertainty cues across a dataset of 5600 articles (each including an early and published version). Change in hedge words is calculated by subtracting the early version's number of predicted hedges from the published version's number of predicted hedges.
### Articles + Other Features
Dataset is made up of 5,600 articles each consisting of an earliest version and published version. All published versions are from 1 of 16 top economics, accounting, or finance journals. Using a combination of two computer vision tools (*add detail here*), PDFs were parsed into JSON format and abstracts, introduction, footnotes, and conclusions were extracted from both versions of all articles. Using 4 different [gendering services](https://github.com/ek8terina/Gendering) we infer the gender of article author(s) from first and secondnames. An article's gender is then the average gender of all authors (1 = male, 0 = female). Article gender and BERT embeddings of each section make up that section's model features to predict the regressant: section's change in hedging (published section - original section). 
## Method
### Training
...
### Results
...
### Prediction
...
## Demo
...
