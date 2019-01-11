from nltk.corpus import wordnet
import numpy as np
import nltk
from time import process_time

def synonyme_vec(word, pos, emb):
    syn_vec=[]
    alt_syn_vec=[]
    synset_pos=''
    if 'N' in pos:
        synset_pos=wordnet.NOUN
    elif 'V' in pos:
        synset_pos=wordnet.VERB
    elif 'J' in pos:
        synset_pos=wordnet.ADJ
    synset_list = wordnet.synsets(word)

    for syn_list in synset_list:
        if syn_list.pos() == synset_pos:
            
            [syn_vec.append(emb[syn]) for syn in syn_list.lemma_names() if syn in emb]
        else:
            [alt_syn_vec.append(emb[syn]) for syn in syn_list.lemma_names() if syn in emb]
    if len(syn_vec) > 1:
        return syn_vec[0]
    elif len(alt_syn_vec) > 1:
        return np.mean(alt_syn_vec, axis=0) 
    else:
        return np.zeros(300, dtype="float32")

def close_distance_word(word, distance_dic, embeddings, pos=None):
#    distance_words = map(lambda x:(x[0],nltk.edit_distance(x,word)), embeddings)
    if word in distance_dic:
        new_word, edit_score = distance_dic[word] 
    else:
        new_word = word
        print(new_word) 
    if new_word in embeddings:
        return embeddings[new_word]
    else:
        return np.zeros(300, dtype="float32")




synonyme_vec('word', 'J', {'word': 2})