import string
import numpy as np
import pandas as pd
import seaborn as sns
from gensim import matutils
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import evaluation_methods as ev
import missing_words_methods as mwm

punctation = list(string.punctuation)
stopWords = stopwords.words('english')


def preprocess_pipeline(sentence):
    word_token = [word.lower() for word in word_tokenize(sentence) 
    if 
    word.lower() not in stopWords and
    word.lower() not in punctation]
    word_token = pos_tag(word_token)
    return word_token

def missing_words_per_sentence(sentence, emb):
    missing_words=[]
    for word, _ in sentence:
        if word not in emb:
            missing_words.append(word)
    return len(missing_words)

def sentence_similarity(emb,sentence_1,sentence_2):
    missing_token=[]
    v1 = []
    v2 = []
    for word in sentence_1:
        try:
            v1.append(emb[word.lower()])
        except KeyError:
            if word.lower() not in missing_token:
                missing_token.append(word.lower())

    for word in sentence_2:
        try:
            v2.append(emb[word.lower()])
        except KeyError:
            if word.lower() not in missing_token:
                missing_token.append(word.lower())
    return np.dot(matutils.unitvec(np.array(v1).mean(axis=0), norm='l2'), matutils.unitvec(np.array(v2).mean(axis=0), norm='l2')), len(v1)/len(sentence_1), len(v2)/len(sentence_2)


def emb_vector_collection(sentence, emb, pos_filter=None, missing_word_strat=None, edit_distance_dic=None, jaccard_distance_dic=None):
    count_pos=0
    count_use=0
    sentence_vec=[]
    missing_token=[]

    if pos_filter != None:
        for word, pos, use in sentence:
            if word in emb and use:
                if pos_filter not in pos:
                    sentence_vec.append(emb[word])
                    count_use+=1
                if pos_filter in pos:
                    sentence_vec.append(mwm.select_missing_word_strat(word=word,emb=emb,
                strat=missing_word_strat, pos=pos, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic))
                    count_pos+=1
            else:
                missing_token.append(word)
    else:
        for word, pos, use in sentence:
            if use and word in emb:
                sentence_vec.append(emb[word])
                count_pos+=1
                count_use+=1
            if use and word not in emb:
                sentence_vec.append(mwm.select_missing_word_strat(word=word,emb=emb,
                strat=missing_word_strat, pos=pos, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic))
            if not use:
               sentence_vec.append(mwm.select_missing_word_strat(word=word,emb=emb,
                strat=missing_word_strat, pos=pos, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic))

    return count_pos,sentence_vec, missing_token, count_use


def rnd_sentence_similarity(emb,sentence_1,sentence_2, size, percent_mode=True, pos_filter=None, method='0-vector', edit_distance_dic=None, jaccard_distance_dic=None):
    new_sent1, new_sent2 = ev.get_random_percentage_lists(sentence_1, sentence_2, size)
    count_filter_pos_s1, v2, _, s2_use_count = emb_vector_collection(new_sent2, emb, pos_filter, missing_word_strat=method, edit_distance_dic=edit_distance_dic,jaccard_distance_dic=jaccard_distance_dic)
    count_filter_pos_s2, v1, _,s1_use_count = emb_vector_collection(new_sent1, emb, pos_filter, missing_word_strat=method, edit_distance_dic=edit_distance_dic,jaccard_distance_dic=jaccard_distance_dic)
    if not sentence_1 or not sentence_2:
    
        print(sentence_2)
        return 0, 1, 1
    try:
        similiarity_score = np.dot(matutils.unitvec(np.mean(v1,axis=0), norm='l2'), matutils.unitvec(np.mean(v2,axis=0), norm='l2'))
    except (TypeError, ValueError):
        similiarity_score =0
    if(pos_filter != None):
        return similiarity_score, s1_use_count/len(sentence_1), s2_use_count/len(sentence_2), count_filter_pos_s1/len(sentence_1), count_filter_pos_s2/len(sentence_2)
    else:
        return similiarity_score, s1_use_count/len(sentence_1), s2_use_count/len(sentence_2)

        
