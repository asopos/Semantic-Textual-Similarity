import csv
import numpy as np
from scipy.stats import pearsonr
from gensim import matutils
import math
import string
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import pandas as pd
import embedding as emb
import evaluation as ev
import warnings

warnings.filterwarnings("ignore")
sentences = []

stemmer = SnowballStemmer("english")
punctation = list(string.punctuation)
stopWords = set(stopwords.words('english'))


google_wv = emb.get_word_embeddings('Word Embeddings\\dev_data_embeddings.txt')
#google_wv = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')

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
    return word_token

print(dev_data.head())
dev_data['sentence_1']=dev_data.apply(lambda row: preprocess_pipeline(row['sentence_1']), axis=1)
dev_data['sentence_2']=dev_data.apply(lambda row: preprocess_pipeline(row['sentence_2']), axis=1)



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
    print(missing_token)
    return np.dot(matutils.unitvec(np.array(v1).mean(axis=0), norm='l2'), matutils.unitvec(np.array(v2).mean(axis=0), norm='l2'))

def corpus_similarity(emb, sentences_1, sentences_2, gold_values, target_path):
    missing_token=[]
    pred_values=[]
    for sentence_1, sentence_2 in zip(sentences_1,sentences_2):
        v1=[]
        v2=[]
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
        pred_values.append(np.dot(matutils.unitvec(np.array(v1).mean(axis=0), norm='l2'), matutils.unitvec(np.array(v2).mean(axis=0), norm='l2')))
    with open(target_path,'w',encoding='utf-8') as f:
        for token in missing_token:
            f.write(str(token) + '\n')
    return pearsonr(pred_values, gold_values)

p_correlation, p_value = corpus_similarity(google_wv, dev_data['sentence_1'].tolist(), dev_data['sentence_2'].tolist(), dev_data['gold_value'].tolist(), 'missing_token.txt')
print(p_correlation*100)


    

#dev_data['pred_value']=dev_data.apply(lambda row: sentence_similarity(google_wv,row['sentence_1'],row['sentence_2']), axis=1)

#p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
print(dev_data.head())