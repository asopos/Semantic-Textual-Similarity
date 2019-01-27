import numpy as np
from nltk.tag import pos_tag
from gensim import matutils
from scipy.stats import pearsonr

def get_word_embeddings(path):
    temp_embeddings={}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split()
            if len(line)>2:
                try:
                    temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
                except ValueError:
                    for index, value in enumerate(line[1:]):
                        if value in '1/5':
                            line[index] = 0.5
                    temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
    return temp_embeddings 

def load_words_distance_dict(source_path):
    word_distance_dict={}
    with open(source_path, encoding='utf-8') as f:
        for line in f:
            line = line.split()
            word_distance_dict[line[0]] = (line[1],line[2])
    return word_distance_dict

