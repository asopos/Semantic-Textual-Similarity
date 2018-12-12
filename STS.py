import csv
import numpy as np
from scipy.stats import pearsonr
from gensim import matutils
import math
import string
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from gensim.models import fasttext
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import pandas as pd
import embedding as emb
import evaluation as ev
import warnings
import random

warnings.filterwarnings("ignore")
sentences = []

stemmer = SnowballStemmer("english")
punctation = list(string.punctuation)
stopWords = set(stopwords.words('english'))


#google_wv = emb.get_word_embeddings('Word Embeddings\\dev_embeddings_googleNews_word2vec')
#google_wv_g = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
glove_wv = emb.get_word_embeddings('Word Embeddings\\dev_embeddings_wiki_glove.txt')

dev_data =pd.read_csv(
    filepath_or_buffer='stsbenchmark\\sts-dev.csv',
#    filepath_or_buffer='./test.csv',
    quoting=csv.QUOTE_NONE,
    sep='\t',
    encoding='utf8',
    header=None,
    usecols=[4,5,6]
    )
dev_data.columns = ['gold_value','sentence_1','sentence_2']

def preprocess_pipeline(sentence):
    word_token = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stopWords]
    word_token = [word for word in word_token if word not in punctation]
    return word_token

dev_data['sentence_1']=dev_data.apply(lambda row: preprocess_pipeline(row['sentence_1']), axis=1)
dev_data['sentence_2']=dev_data.apply(lambda row: preprocess_pipeline(row['sentence_2']), axis=1)

print(dev_data.tail())

def sentence_coverage(word_tokens, emb):
    count=0
    for token in word_tokens:
        if token in emb:
            count+=1
    return count/len(word_tokens)

def sentences_cov(word_token_1, word_token_2, emb):
    cov_1=sentence_coverage(word_token_1,emb)
    cov_2=sentence_coverage(word_token_2,emb)
    return (cov_1+cov_2)/2



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

def mod_sentence_similarity(emb,sentence_1,sentence_2, seperate):
    missing_token=[]
    v1 = []
    v2 = []
    s_1_sep=0
    s_2_sep=0
    for word in sentence_1:
        try:
            if s_1_sep % seperate==0:
                v1.append(emb[word.lower()])
                s_1_sep+=1
            else:
                s_1_sep+=1
        except KeyError:
            if word.lower() not in missing_token:
                missing_token.append(word.lower())

    for word in sentence_2:
        try:
            if s_2_sep % seperate == 0:
                v2.append(emb[word.lower()])
                s_2_sep+=1
            else:
                s_2_sep+=1
        except KeyError:
            if word.lower() not in missing_token:
                missing_token.append(word.lower())
    
    try:
        similiarity_score = np.dot(matutils.unitvec(np.array(v1).mean(axis=0), norm='l2'), matutils.unitvec(np.array(v2).mean(axis=0), norm='l2'))
    except TypeError:
        similiarity_score =0
    return similiarity_score, len(v1)/len(sentence_1), len(v2)/len(sentence_2)

def emb_vector_collection(sentence, emb):
    sentence_vec=[]
    missing_token=[]
    for word in sentence:
        try:
            sentence_vec.append(emb[word.lower()])
        except KeyError:
            missing_token.append(word.lower())
    return sentence_vec


def rnd_sentence_similarity(emb,sentence_1,sentence_2, size):
    v2 = emb_vector_collection(sentence_2, emb)
    v1 = emb_vector_collection(sentence_1, emb)
    
    size_1= math.ceil(size*len(sentence_1))
    size_2= math.ceil(size*len(sentence_2))
    if len(v1) >= size_1:
        v_1 = random.sample(v1, size_1)
    else:
        v_1 = v1
    if len(v2) >= size_2:
        v_2 = random.sample(v2, size_2)
    else:
        v_2 = v2
    try:
        similiarity_score = np.dot(matutils.unitvec(np.array(v_1).mean(axis=0), norm='l2'), matutils.unitvec(np.array(v_2).mean(axis=0), norm='l2'))
    except TypeError:
        similiarity_score =0
    return similiarity_score, len(v_1)/len(sentence_1), len(v_2)/len(sentence_2)

def corpus_similarity(emb, sentences_1, sentences_2, gold_values, target_path):
    missing_token=[]
    pred_values=[]
    for sentence_1, sentence_2 in zip(sentences_1,sentences_2):
        v1=[]
        v2=[]
        for word in sentence_1:
            try:
                v1.append(emb[word])
            except KeyError:
                if word.lower() not in missing_token:
                    missing_token.append(word)
        for word in sentence_2:
            try:
                v2.append(emb[word])
            except KeyError:
                if word.lower() not in missing_token:
                    missing_token.append(word)
        pred_values.append(np.dot(matutils.unitvec(np.array(v1).mean(axis=0), norm='l2'), matutils.unitvec(np.array(v2).mean(axis=0), norm='l2')))
    return pearsonr(pred_values, gold_values)

#p_correlation, p_value = corpus_similarity(google_wv, dev_data['sentence_1'].tolist(), dev_data['sentence_2'].tolist(), dev_data['gold_value'].tolist(), 'missing_token.txt')


#p_correlation, p_value = corpus_similarity(google_wv_g, dev_data['sentence_1'].tolist(), dev_data['sentence_2'].tolist(), dev_data['gold_value'].tolist(), 'missing_token.txt')
    

dev_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=dev_data.apply(
    lambda row: pd.Series(rnd_sentence_similarity(glove_wv,row['sentence_1'],row['sentence_2'], 1)), axis=1)



#dev_data['cov_avrg']=dev_data.apply(lambda row: sentences_cov(row['sentence_1'],row['sentence_2'], google_wv), axis=1)

p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
#print(dev_data.tail())
#print(p_correlation)
avrg_cov_s1 = sum(dev_data['cov_avrg_s1'].tolist())/len(dev_data['cov_avrg_s1'].tolist())
avrg_cov_s2 = sum(dev_data['cov_avrg_s2'].tolist())/len(dev_data['cov_avrg_s2'].tolist())

#print((avrg_cov_s1 + avrg_cov_s2)/2)
test=0.1
min=0
max=0
i=0
while test < 1.1:
    result_list=[]
    i=0
    while i<1001:
        dev_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=dev_data.apply(
        lambda row: pd.Series(rnd_sentence_similarity(glove_wv,row['sentence_1'],row['sentence_2'], test)), axis=1)
        
        p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
        if i ==0:
            min = p_correlation
            max = p_correlation
        if p_correlation < min:
            min = p_correlation
        if p_correlation > max:
            max = p_correlation
        result_list.append(p_correlation)
    #    print(str(i)+ '- ' + str(p_correlation)+" - min: "+ str(min) + ' max: '+ str(max) )
        i+=1
    avrg_cov_s1 = sum(dev_data['cov_avrg_s1'].tolist())/len(dev_data['cov_avrg_s1'].tolist())
    avrg_cov_s2 = sum(dev_data['cov_avrg_s2'].tolist())/len(dev_data['cov_avrg_s2'].tolist())
    print('--------------- ' + str(test))
    print('Min: '+str(min*100), 'Max: '+ str(max*100))
    print('Avrg: '+ str(np.mean(result_list)*100))
    print('Avr_Coverage: '+ str((avrg_cov_s1 + avrg_cov_s2)/2))
    test = test + 0.1

#print(min, max)
#print(np.mean(result_list))