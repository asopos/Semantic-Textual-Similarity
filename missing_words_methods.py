from nltk.corpus import wordnet
import numpy as np
import nltk
from time import process_time

def wordnet_vec(word, pos, emb, useHypernyms=False):
    wordnet_vec=[]
    alt_wordnet_vec=[]
    synset_pos=''
    if 'N' in pos:
        synset_pos=wordnet.NOUN
    elif 'V' in pos:
        synset_pos=wordnet.VERB
    elif 'J' in pos:
        synset_pos=wordnet.ADJ
    
    synset_list = wordnet.synsets(word)
    if useHypernyms == False:
        for syn_list in synset_list:
            if syn_list.pos() == synset_pos:
                [wordnet_vec.append(emb[syn]) for syn in syn_list.lemma_names() if syn in emb]
            else:
                [alt_wordnet_vec.append(emb[syn]) for syn in syn_list.lemma_names() if syn in emb]
    else:
        for syn_list in synset_list:
            hyper_list = syn_list.hypernyms()
            if syn_list.pos() == synset_pos:
                [wordnet_vec.append(emb[hyper.lemmas()[0].name()]) for hyper in hyper_list if hyper.lemmas()[0].name() in emb]
            else:
                [alt_wordnet_vec.append(emb[hyper.lemmas()[0].name()]) for hyper in hyper_list if hyper.lemmas()[0].name() in emb]
    if len(wordnet_vec) > 1:
        return wordnet_vec[0]
    elif len(alt_wordnet_vec) > 1:
        return np.mean(alt_wordnet_vec, axis=0) 
    else:
        return np.zeros(300, dtype="float32")



def close_distance_word(word, distance_dic, embeddings, pos=None):
    if word in distance_dic:
        new_word, edit_score = distance_dic[word]
        if new_word in embeddings:
            return embeddings[new_word]
    else:
        print(word)
        return np.zeros(300, dtype="float32")

def random_vector():
    return np.random.random(300)


def select_missing_word_strat(word,emb, strat='0-vector', edit_distance_dic=None, pos=None, jaccard_distance_dic=None):
    if strat == '0-vector':
        return np.zeros(300, dtype="float32")
    elif strat == 'synonym':
        return wordnet_vec(word, pos, emb)
    elif strat == 'edit_distance':
        return close_distance_word(word, edit_distance_dic, emb)
    elif strat =='random':
        return random_vector()
    elif strat =='hypernym':
        return wordnet_vec(word, pos, emb, True)
    elif strat =='jaccard_distance':
        return close_distance_word(word, jaccard_distance_dic, emb)