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
from nltk import pos_tag, pos_tag_sents
from nltk.tag.stanford import StanfordPOSTagger
import pandas as pd
import embedding as emb
import evaluation as ev
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
java_path = "C:\Program Files (x86)\Common Files\Oracle\Java\javapath\java.exe"
os.environ['JAVAHOME'] = java_path

st_pos_tagger = StanfordPOSTagger('english-left3words-distsim.tagger', 'C:\\Users\\Rene\\Documents\\stanford-postagger-2018-10-16\\stanford-postagger.jar')

warnings.filterwarnings("ignore")
sentences = []

stemmer = SnowballStemmer("english")
punctation = list(string.punctuation)
stopWords = set(stopwords.words('english'))

#fastText_emb = FastText.load_fasttext_format('Word Embeddings\\wiki.en.bin', 'utf-8')

#fastText_emb = emb.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_fasttext.txt')
google_wv = emb.get_word_embeddings('Word Embeddings\\train_embeddings_googleNews_word2vec.txt')
#google_wv_g = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
#glove_wv = emb.get_word_embeddings('Word Embeddings\\dev_embeddings_wiki_glove.txt')

dev_data =pd.read_csv(
    filepath_or_buffer='stsbenchmark\\sts-train.csv',
#    filepath_or_buffer='./test.csv',
    quoting=csv.QUOTE_NONE,
    sep='\t',
    encoding='utf8',
    header=None,
    usecols=[4,5,6]
    )
dev_data.columns = ['gold_value','sentence_1','sentence_2']

def preprocess_pipeline(sentence):
    word_token = [word.lower() for word in word_tokenize(sentence) 
    if word.lower() not in stopWords 
    and word.lower() not in punctation]
    #word_token = pos_tag(word_token)
    #if word.lower() not in stopWords
    return word_token


dev_data[['sentence_1', 'sentence_2']]=dev_data.apply(
    lambda row: pd.Series([preprocess_pipeline(row['sentence_1']),preprocess_pipeline(row['sentence_2'])]), axis=1)


dev_data[['sentence_len_1', 'sentence_len_2']]=dev_data.apply(
    lambda row: pd.Series([len(row['sentence_1']),len(row['sentence_2'])]), axis=1)

#dev_data[['sentence_1', 'sentence_2']] = pd.Series(st_pos_tagger.tag_sents(dev_data['sentence_1'].tolist()), st_pos_tagger.tag_sents(dev_data['sentence_2'].tolist()))

dev_data['sentence_1'] = pos_tag_sents(dev_data['sentence_1'].tolist())
dev_data['sentence_2'] = pos_tag_sents(dev_data['sentence_2'].tolist())

#print(dev_data.tail())

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
        for word, pos in sentence:
            if word in emb:
                if pos != pos_filter:
                    sentence_vec.append(emb[word])
                if pos == pos_filter:
                    count_pos+=1
            else:
                missing_token.append(word)
    else:
        for word, pos in sentence:
            if word in emb:
                sentence_vec.append(emb[word])
                count_pos+=1
            else:
                missing_token.append(word)

    return count_pos,sentence_vec, missing_token


def rnd_sentence_similarity(emb,sentence_1,sentence_2, size, percent_mode=False, pos_filter=None):
    count_filter_pos_s1, v2, _ = emb_vector_collection(sentence_2, emb, pos_filter)
    count_filter_pos_s2, v1, _ = emb_vector_collection(sentence_1, emb, pos_filter)
    if percent_mode is True:
        size_1, size_2= round(size*len(sentence_1)), round(size*len(sentence_2))
        if size_1 ==0:
            size_1=1
        if size_2 ==0:
            size_2=1
    else:
        size_1, size_2 = size, size
    if len(v1) > size_1:
        v_1 = random.sample(v1, size_1)
    else:
        v_1 = v1
    if len(v2) > size_2:
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


def get_pos_distribution(sentences):
    pos_list = []
    word_count = 0
    for sentence in sentences:
        for _ , pos in sentence:
            # if 'J' in pos:
            #     pos='J'
            # elif 'N' in pos:
            #     pos='N'
            # elif 'V' in pos:
            #     pos='V'
            # elif 'CD' in pos:
            #     pos='C'
            #else:
                #pos='egal'
            pos_list.append(pos)
            word_count+=1
    c = Counter(pos_list)
    percent_list = [(i, c[i] / word_count * 100.0) for i, count in c.most_common()]
    return percent_list
    
def get_sentence_size_distribution(sentences):
    sentence_size_list=[]
    for sentence in sentences:
        sentence_size_list.append(len(sentence))

    c = Counter(sentence_size_list)
    percent_list = [(i, c[i] / len(sentences) * 100.0) for i, count in c.most_common()]
    return np.average(sentence_size_list), percent_list

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

fmri = sns.load_dataset("fmri")
#print(fmri)
def  pos_filter_similarity(data_frame,emb,pos_list):
    noune_list =[]
    verb_list = []
    adj_list = []
    for pos in pos_list:
        data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2', 'pos_cov_avr_s1', 'pos_cov_avr_s2']]=data_frame.apply(
        lambda row: pd.Series(
            rnd_sentence_similarity(emb ,row['sentence_1'],row['sentence_2'], 1, pos_filter=pos, percent_mode=True)), axis=1)
        
        p_correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
        
        avrg_cov = np.mean(data_frame[['cov_avrg_s1','cov_avrg_s2']].values)
        avrg_pos_cov = np.mean(data_frame[['pos_cov_avr_s1','pos_cov_avr_s2']].values)
        
        print(pos+' --------------------' )
        print('Pearson - Correlation: '+ str(p_correlation*100))
        print('Avr_Coverage: '+str(avrg_cov))
        print('POS_Coverage: ' + str(avrg_pos_cov))
    return 
    

dev_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=dev_data.apply(lambda row: pd.Series(rnd_sentence_similarity(google_wv,row['sentence_1'],row['sentence_2'], 1, percent_mode=True)), axis=1)

p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
print(p_correlation)

ax = sns.regplot(dev_data['pred_value'],dev_data['gold_value'], scatter_kws={"color": "black", 'alpha':0.1}, line_kws={"color": "red"})
ax.set(ylim=(0, 5))
plt.show()
avrg_cov = np.mean(dev_data[['cov_avrg_s1','cov_avrg_s2']].values)

#avrg_pos_cov_s1 = sum(dev_data['pos_cov_avr_s1'].tolist())/len(dev_data['pos_cov_avr_s1'].tolist())
#avrg_pos_cov_s2 = sum(dev_data['pos_cov_avr_s2'].tolist())/len(dev_data['pos_cov_avr_s2'].tolist())

print(avrg_cov)
#print((avrg_pos_cov_s1 + avrg_pos_cov_s2)/2)


def evaluate_rnd_coverage_emb(emb, data_frame, iterations, percent_mode):
    if percent_mode is True:
        size=0.05
    else:
        size =1
    _, axes = plt.subplots(5, 4)
    axes = axes.flatten()
    for ax in axes:
        result_list=[]
        i=0
        while i < iterations:
            data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=data_frame.apply(
                lambda row: pd.Series(rnd_sentence_similarity(emb,row['sentence_1'],row['sentence_2'], size=size, percent_mode=percent_mode)), axis=1)
            p_correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
            result_list.append(round(p_correlation*100,2))
            i+=1
        avr_cov = np.mean(data_frame[['cov_avrg_s1','cov_avrg_s2']].values)
        avr_pearson = np.mean(result_list)
        min_pearson = np.min(result_list)
        max_pearson = np.max(result_list)
        print('--------------- ' + str(size))
        print('Min: '+str(min_pearson), 'Max: '+ str(max_pearson))
        print('Avrg: '+ str(avr_pearson))
        print('Avr_Coverage: '+ str(avr_cov))
        try:
            sns.distplot(result_list,norm_hist=True,ax=ax)
        except np.linalg.linalg.LinAlgError:
            print(result_list) 
        if percent_mode is True:
            size+=0.05
        else:
            size+=1
    plt.show()
    return None

sentences = dev_data['sentence_1'].tolist() + dev_data['sentence_2'].tolist()

test_df = dev_data[abs(dev_data.sentence_len_1 - dev_data.sentence_len_2) < 1]

print(test_df.tail())

pos_percent =get_pos_distribution(sentences)
_, sent_percent = get_sentence_size_distribution(sentences)
pos_list = [pos for pos, _ in pos_percent]
#evaluate_rnd_coverage_emb(google_wv, test_df, 100, percent_mode=True)

pos_filter_similarity(emb=google_wv, data_frame=dev_data,pos_list=pos_list)

