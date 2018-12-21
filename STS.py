import csv
import numpy as np
from scipy.stats import pearsonr
from gensim import matutils
import math
import string
from nltk.stem import SnowballStemmer
from gensim.models import KeyedVectors
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
import pandas as pd
import embedding as emb
import evaluation as ev
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sentences = []

stemmer = SnowballStemmer("english")
punctation = list(string.punctuation)
stopWords = set(stopwords.words('english'))

#fastText_emb = FastText.load_fasttext_format('Word Embeddings\\wiki.en.bin', 'utf-8')

fastText_emb = emb.get_word_embeddings('Word Embeddings\\dev_embeddings_wiki_fasttext.txt')
#google_wv = emb.get_word_embeddings('Word Embeddings\\dev_embeddings_googleNews_word2vec.txt')
#google_wv_g = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
#glove_wv = emb.get_word_embeddings('Word Embeddings\\dev_embeddings_wiki_glove.txt')

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
    #if word.lower() not in stopWords
    word_token = [word for word in word_token if word not in punctation]
    return word_token

dev_data[['sentence_1', 'sentence_2']]=dev_data.apply(
    lambda row: pd.Series([preprocess_pipeline(row['sentence_1']),preprocess_pipeline(row['sentence_2'])]), axis=1)

print(dev_data.tail())

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


def emb_vector_collection(sentence, emb, pos_filter=None):
    count_pos=0
    sentence_vec=[]
    missing_token=[]
    if pos_filter != None:
        pos_tags = pos_tag(sentence)
        for word, pos in pos_tags:
            try:
                if pos_filter == None:
                    sentence_vec.append(emb[word.lower()])
                if pos_filter != None and pos != pos_filter:
                    sentence_vec.append(emb[word.lower()])
                if pos == pos_filter:
                    count_pos+=1
            except KeyError:
                missing_token.append(word.lower())
    else:
        for word in sentence:
            try:
                sentence_vec.append(emb[word.lower()])
                count_pos+=1
            except KeyError:
                missing_token.append(word.lower())

    return count_pos,sentence_vec, missing_token


def rnd_sentence_similarity(emb,sentence_1,sentence_2, percent, pos_filter=None):
    count_filter_pos_s1, v2, miss_s2 = emb_vector_collection(sentence_2, emb, pos_filter)
    count_filter_pos_s2, v1, miss_s1 = emb_vector_collection(sentence_1, emb, pos_filter)
    
    size_1= math.ceil(percent*len(sentence_1))
    size_2= math.ceil(percent*len(sentence_2))
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
    if(pos_filter != None):
        return similiarity_score, len(v_1)/len(sentence_1), len(v_2)/len(sentence_2), count_filter_pos_s1/len(sentence_1), count_filter_pos_s2/len(sentence_2)
    else:
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

pos_list=['CC', 'CD','DT', 'EX','FW','IN','JJ', 'JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
noun_list=['NN','NNS','NNP','NNPS']
verb_list=['VB','VBD','VBG','VBN','VBP','VBZ']
adj_list=['JJ', 'JJR','JJS']


# for pos in verb_list:
#     dev_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2', 'pos_cov_avr_s1', 'pos_cov_avr_s2']]=dev_data.apply(
#         lambda row: pd.Series(rnd_sentence_similarity(fastText_emb,row['sentence_1'],row['sentence_2'], 1, pos)), axis=1)
#     p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
#     avrg_cov_s1 = sum(dev_data['cov_avrg_s1'].tolist())/len(dev_data['cov_avrg_s1'].tolist())
#     avrg_cov_s2 = sum(dev_data['cov_avrg_s2'].tolist())/len(dev_data['cov_avrg_s2'].tolist())

#     avrg_pos_cov_s1 = sum(dev_data['pos_cov_avr_s1'].tolist())/len(dev_data['pos_cov_avr_s1'].tolist())
#     avrg_pos_cov_s2 = sum(dev_data['pos_cov_avr_s2'].tolist())/len(dev_data['pos_cov_avr_s2'].tolist())
#     print(pos+' --------------------' )
#     print('Pearson - Correlation: '+ str(p_correlation*100))
#     print('Avr_Coverage: '+str((avrg_cov_s1 + avrg_cov_s2)/2))
#     print('POS_Coverage: ' + str((avrg_pos_cov_s1 + avrg_pos_cov_s2)/2))
dev_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=dev_data.apply(lambda row: pd.Series(rnd_sentence_similarity(fastText_emb,row['sentence_1'],row['sentence_2'], 1)), axis=1)

p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
print(dev_data.head())
print(p_correlation)

sns.regplot(dev_data['gold_value'],dev_data['pred_value'])
plt.show()
avrg_cov_s1 = sum(dev_data['cov_avrg_s1'].tolist())/len(dev_data['cov_avrg_s1'].tolist())
avrg_cov_s2 = sum(dev_data['cov_avrg_s2'].tolist())/len(dev_data['cov_avrg_s2'].tolist())

#avrg_pos_cov_s1 = sum(dev_data['pos_cov_avr_s1'].tolist())/len(dev_data['pos_cov_avr_s1'].tolist())
#avrg_pos_cov_s2 = sum(dev_data['pos_cov_avr_s2'].tolist())/len(dev_data['pos_cov_avr_s2'].tolist())

print((avrg_cov_s1 + avrg_cov_s2)/2)
#print((avrg_pos_cov_s1 + avrg_pos_cov_s2)/2)


def evaluate_rnd_coverage_emb(emb, data_frame, iterations, plt):
    percent=0.1
    while percent <1.1:
        result_list=[]
        i=0
        while i < iterations:
            data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=data_frame.apply(
                lambda row: pd.Series(rnd_sentence_similarity(emb,row['sentence_1'],row['sentence_2'], percent)), axis=1)
            p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
            if i ==0:
                min = p_correlation
                max = p_correlation
            if p_correlation < min:
                min = p_correlation
            if p_correlation > max:
                max = p_correlation
            result_list.append(round(p_correlation*100,2))
            print(i)
            i+=1
        avrg_cov_s1 = sum(dev_data['cov_avrg_s1'].tolist())/len(dev_data['cov_avrg_s1'].tolist())
        avrg_cov_s2 = sum(dev_data['cov_avrg_s2'].tolist())/len(dev_data['cov_avrg_s2'].tolist())
        print('--------------- ' + str(percent))
        print('Min: '+str(min*100), 'Max: '+ str(max*100))
        print('Avrg: '+ str(np.mean(result_list)))
        print('Avr_Coverage: '+ str((avrg_cov_s1 + avrg_cov_s2)/2))
        percent = percent + 0.1
        sns.distplot(result_list,norm_hist=True)
#        plt.hist(result_list, normed=True, bins=10)
#        plt.title('Avr_Coverage: '+ str((avrg_cov_s1 + avrg_cov_s2)/2)+'Min: '+str(min*100)+  ' Max: '+ str(max*100) + 'Avrg: '+ str(np.mean(result_list)))
#        plt.xlabel("p-correlation")
#        plt.ylabel("Frequency")
        plt.show()
    return None


evaluate_rnd_coverage_emb(fastText_emb, dev_data, 100, plt)