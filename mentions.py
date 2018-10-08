#!/usr/bin/env python
# coding: utf8
# Module for neuralcoref: https://github.com/huggingface/neuralcoref

import spacy
nlp = spacy.load('en_coref_md')

from sklearn.datasets import load_files

# Loading data using sklearn load_files function

data_folder = 'data_structure_for_classification/'
all_data = load_files(data_folder)

# Saving and loading all text documents

documents = all_data.data
        
# Converting bytes-object into string

text = []
for i in documents:
    i = i.decode("utf8")
    text.append(i)
    
# Reference link: https://stackoverflow.com/questions/606191/convert-bytes-to-a-string

# Loading neuralcoref module

nlp = spacy.load('en_coref_md')

# Performing coreference resolution and extracting mention clusters for every text document through a for loop

mentions_list = []
for i in text:
	doc = nlp(i)
	mentions = doc._.coref_clusters
	mentions_list.append(mentions)

# Reference link:  https://github.com/huggingface/neuralcoref

# Saving the output as text files

filenames = all_data.filenames

filenames_list = []
for i in filenames:
    y = str(i) + '_mentions.txt'
    filenames_list.append(y)

for i, y in zip(mentions_list, filenames_list):
    with open(y, 'w', encoding='utf-8') as output:
        output.write(i)

# Reference link: https://stackoverflow.com/questions/6673092/printing-out-elements-of-list-into-separate-text-files-in-python
# Reference link: https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters

