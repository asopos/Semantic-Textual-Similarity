import csv
import numpy as np
from scipy.stats import pearsonr, spearmanr
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
import missing_words_methods as mwm
import pandas as pd
import embedding_helper as emb
import evaluation as ev
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

warnings.filterwarnings("ignore")
sentences = []

punctation = list(string.punctuation)
stopWords = stopwords.words('english')

#fastText_emb = FastText.load_fasttext_format('Word Embeddings\\wiki.en.bin', 'utf-8')

#fastText_emb_no_char = emb.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_fasttext_more.txt')
fastText_emb = emb.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_fasttext_word.txt')
#google_wv = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
google_wv = emb.get_word_embeddings('Word Embeddings\\train_embeddings_googleNews_word2vec.txt')
#google_wv_g = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
#glove_wv = emb.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_glove_official.txt')

#------------------------------
google_vw_edit_distance = emb.load_words_distance_dict('string_distance_mapping\\all_words_train_edit_distance_word2vec.txt')
google_vw_jac_distance = emb.load_words_distance_dict('string_distance_mapping\\train_jaccard_distance_word2vec.txt')
#fasttext_vw_edit_distance = emb.load_words_distance_dict('string_distance_mapping\\test_train_edit_distance_fasttext.txt')
#fasttext_vw_jac_distance = emb.load_words_distance_dict('string_distance_mapping\\test_train_jaccard_distance_fasttext.txt')
#glove_wv_edit_distance = emb.load_words_distance_dict('string_distance_mapping\\test_train_edit_distance_glove_official.txt')
#glove_wv_jac_distance = emb.load_words_distance_dict('string_distance_mapping\\test_train_jaccard_distance_glove_official.txt')

emb_of_choice=google_wv
edit_dist_of_choice= google_vw_edit_distance
jac_dist_of_choice = google_vw_jac_distance

dev_data =pd.read_csv(
    filepath_or_buffer='stsbenchmark\\sts-train.csv',
    quoting=csv.QUOTE_NONE,
    sep='\t',
    encoding='utf8',
    header=None,
    usecols=[4,5,6]
    )
dev_data.columns = ['gold_value','sentence_1','sentence_2']

def preprocess_pipeline(sentence):
    word_token = [word.lower() for word in word_tokenize(sentence) 
    if 
    word.lower() not in stopWords and
    word.lower() not in punctation]
    #word_token = pos_tag(word_token)
    #if word.lower() not in stopWords
    return word_token

def missing_words_per_sentence(sentence, emb):
    missing_words=[]
    for word in sentence:
        if word not in emb:
            missing_words.append(word)
    return len(missing_words)

dev_data[['sentence_1', 'sentence_2']]=dev_data.apply(
    lambda row: pd.Series([preprocess_pipeline(row['sentence_1']),preprocess_pipeline(row['sentence_2'])]), axis=1)

dev_data['missing_emb'] = dev_data.apply(lambda row: missing_words_per_sentence(row['sentence_1']+ row['sentence_2'], emb_of_choice), axis=1)

dev_data[['sentence_len_1', 'sentence_len_2']]=dev_data.apply(
    lambda row: pd.Series([len(row['sentence_1']),len(row['sentence_2'])]), axis=1)

dev_data['sentence_1'] = pos_tag_sents(dev_data['sentence_1'].tolist())
dev_data['sentence_2'] = pos_tag_sents(dev_data['sentence_2'].tolist())

print(dev_data.head())

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
    sentence_emp_vec=[emb[element[0]] for element in sentence if element[0] in emb]

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
                #count_use+=1
            if not use:
               sentence_vec.append(mwm.select_missing_word_strat(word=word,emb=emb,
                strat=missing_word_strat, pos=pos, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic))
                #sentence_vec.append(np.mean(sentence_emp_vec,axis=0))

    return count_pos,sentence_vec, missing_token, count_use


def rnd_sentence_similarity(emb,sentence_1,sentence_2, size, percent_mode=False, pos_filter=None, method='0-vector', edit_distance_dic=None, jaccard_distance_dic=None):
    new_sent1, new_sent2 = ev.get_random_percentage_lists(sentence_1, sentence_2, size)
    count_filter_pos_s1, v2, _, s2_use_count = emb_vector_collection(new_sent2, emb, pos_filter, missing_word_strat=method, edit_distance_dic=edit_distance_dic,jaccard_distance_dic=jaccard_distance_dic)
    count_filter_pos_s2, v1, _,s1_use_count = emb_vector_collection(new_sent1, emb, pos_filter, missing_word_strat=method, edit_distance_dic=edit_distance_dic,jaccard_distance_dic=jaccard_distance_dic)
    
    try:
        similiarity_score = np.dot(matutils.unitvec(np.mean(v1,axis=0), norm='l2'), matutils.unitvec(np.mean(v2,axis=0), norm='l2'))
    except (TypeError, ValueError):
        similiarity_score =0
    if(pos_filter != None):
        return similiarity_score, s1_use_count/len(sentence_1), s2_use_count/len(sentence_2), count_filter_pos_s1/len(sentence_1), count_filter_pos_s2/len(sentence_2)
    else:
        return similiarity_score, s1_use_count/len(sentence_1), s2_use_count/len(sentence_2)


def get_pos_distribution(sentences):
    pos_list = []
    pos_dict={}
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
    for pos, percent in percent_list:
        pos_dict[pos] = round(percent*100)
#    pos_dataframe = pd.DataFrame.from_dict(pos_dict)
#    g = sns.catplot(data=pos_dataframe, kind="bar")
#    plt.show()
    return percent_list
    
def get_sentence_size_distribution(sentences):
    sentence_size_list=[]
    for sentence in sentences:
        sentence_size_list.append(len(sentence))

    c = Counter(sentence_size_list)
    percent_list = [(i, c[i] / len(sentences) * 100.0) for i, count in c.most_common()]
    return np.average(sentence_size_list), percent_list

pos_list=['CC', 'CD','DT', 'EX','FW','IN','JJ', 'JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
noun_list=['NN','NNS','NNP','NNPS']
verb_list=['VB','VBD','VBG','VBN','VBP','VBZ']
adj_list=['JJ', 'JJR','JJS']

def pos_filter_similarity(data_frame,emb,pos_filter, methods, edit_distance_dic=None, jaccard_distance_dic=None):
    pos_list=[]
    pearson_score=[]
    spearman_score=[]
    method_list=[]
    for method in methods:
        for pos in pos_filter:
            data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2', 'pos_cov_avr_s1', 'pos_cov_avr_s2']]=data_frame.apply(
            lambda row: pd.Series(
                rnd_sentence_similarity(emb ,row['sentence_1'],row['sentence_2'], 1, pos_filter=pos, percent_mode=True, method=method, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic)), axis=1)
            
            p_correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
            s_correlation, _ = spearmanr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
            method_list.append(method)
            pearson_score.append(round(p_correlation*100,2))
            spearman_score.append(round(s_correlation*100,2))
            pos_list.append(pos)

            avrg_cov = np.mean(data_frame[['cov_avrg_s1','cov_avrg_s2']].values)
            avrg_pos_cov = np.mean(data_frame[['pos_cov_avr_s1','pos_cov_avr_s2']].values)
            
            print(pos+' --------------------'+ ': ' + method )
            print('Pearson - Correlation: '+ str(round(p_correlation*100,2)))
            print('Spearman - Correlation: '+ str(round(s_correlation*100,2)))
            print('Avr_Coverage: '+str(avrg_cov))
            print('POS_Coverage: ' + str(avrg_pos_cov))
    df = pd.DataFrame({'Pearson_Score': pearson_score, 'Spearman_Score': spearman_score, 'POS':pos_list, 'Methods': method_list})
    f = sns.barplot(x='POS', y='Pearson_Score',data=df, hue='Methods')
    plt.show()
    f = sns.barplot(x='POS', y='Spearman_Score',data=df, hue='Methods')
    print(df)
    plt.show()
    
def missing_words_similarity(data_frame,emb, edit_distance_dic=None, jaccard_distance_dic=None ,methods=['0-vector']):
    missing_sentence_count=set((data_frame['missing_emb'].tolist()))
    method_list =[]
    sentence_count = []
    aktual_miss=[]
    pearson_score=[]
    spearman_score=[]
    for method in methods:
        for word_miss in missing_sentence_count:
            miss_data = data_frame.query('missing_emb =='+str(word_miss))
            miss_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=miss_data.apply(
                lambda row: pd.Series(rnd_sentence_similarity(emb,row['sentence_1'],row['sentence_2'], 1, percent_mode=True,method=method, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic)), axis=1)
            p_correlation, p_value = pearsonr(miss_data['pred_value'].tolist(), miss_data['gold_value'].tolist())
            s_correlation , s_value = spearmanr(miss_data['pred_value'].tolist(), miss_data['gold_value'].tolist())
            avrg_cov = np.mean(miss_data[['cov_avrg_s1','cov_avrg_s2']].values)
            if len(miss_data.index) > 100:
                sentence_count.append(len(miss_data.index))
                method_list.append(method)
                pearson_score.append(round(p_correlation*100,2))
                spearman_score.append(round(s_correlation*100,2))
                aktual_miss.append(word_miss)
            print('--------------- ' + str(word_miss)+ ': ' + str(method))
            print('Pearson - Correlation: '+ str(p_correlation*100))
            print('Spearman - Correlation: '+ str(s_correlation*100))
            print('Sentence Count: '+ str(len(miss_data.index)))
            print('Avr_Coverage: '+ str(avrg_cov))
    df = pd.DataFrame({'Pearson_Score': pearson_score, 'Spearman_Score': spearman_score, 'Missing_Words':aktual_miss, 'Method': method_list, 'Sentence_Count': sentence_count})
    print(df)
    g = sns.factorplot(x="Missing_Words", y="Pearson_Score", hue='Method', data=df)
    g = sns.factorplot(x="Missing_Words", y="Spearman_Score", hue='Method', data=df)
    plt.show()
    f = sns.barplot(x='Missing_Words', y='Sentence_Count',data=df)
    plt.show()


def evaluation_emb(embeddings, data_frame):
    embedding_list=["fasttext","word2vec", "glove"]
    s_cor = []
    p_cor =[]
    word_cov = []
    for emb in embeddings:
        data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=data_frame.apply(
        lambda row: pd.Series(rnd_sentence_similarity(
            emb,row['sentence_1'],row['sentence_2'], size=1, percent_mode=True)), axis=1)
        p_correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
        s_correlation, _ = spearmanr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
        word_cov.append(round(np.mean(data_frame[['cov_avrg_s1','cov_avrg_s2']].values)*100,1))
        p_cor.append(round(p_correlation*100,1))
        s_cor.append(round(s_correlation*100,1))


    df = pd.DataFrame({'pearson': p_cor, 'spearman': s_cor, 'Word Embeddings': embedding_list , 'Wortabdeckung': word_cov})
    df = df.melt('Word Embeddings', var_name='Korrelation', value_name='Wert')
    print(df)
    sns.barplot(x='Word Embeddings', y='Wert', hue='Korrelation', data=df)
    plt.show()
    


def evaluate_rnd_coverage_emb(emb, data_frame, iterations, percent_mode, methods=None, edit_distance_dic=None, jaccard_distance_dic=None):
    if percent_mode is True:
        size=1
    else:
        size =1
    _, axes = plt.subplots(5, 2)
    axes = axes.flatten()
    result_dict={}
    method_list = [ item for item in methods for _ in range(iterations) ] 
    for method in methods:
        for ax in axes:
            result_list=[]
            i=0
            while i < iterations:
                data_frame[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=data_frame.apply(
                    lambda row: pd.Series(rnd_sentence_similarity(
                        emb,row['sentence_1'],row['sentence_2'], size=size, percent_mode=percent_mode, method=method, edit_distance_dic=edit_distance_dic, jaccard_distance_dic=jaccard_distance_dic)), axis=1)
                p_correlation, _ = pearsonr(data_frame['pred_value'].tolist(), data_frame['gold_value'].tolist())
                result_list.append(round(p_correlation*100,2))
                i+=1
            avr_cov = np.mean(data_frame[['cov_avrg_s1','cov_avrg_s2']].values)
            avr_pearson = np.mean(result_list)
            min_pearson = np.min(result_list)
            max_pearson = np.max(result_list)
            print('--------------- ' + str(size)+ ': '+ method)
            print('Min: '+str(min_pearson), 'Max: '+ str(max_pearson))
            print('Avrg: '+ str(avr_pearson))
            print('Avr_Coverage: '+ str(avr_cov))

            if round(size*100,0) in result_dict:
                result_dict[round(size*100,0)] = result_dict[round(size*100,0)] + result_list
            else:
                result_dict[round(size*100,0)]=result_list

            try:
                if len(result_list) > 1:
                    ax.set_title(str(round(avr_cov*100,0)))
                    sns.distplot(result_list,norm_hist=True,ax=ax,label=size)
            except np.linalg.linalg.LinAlgError:
                print(result_list) 
            if percent_mode is True:
                size=round(size-0.1,1)
            else:
                size+=1
        plt.show()
        size=1
    result_dict['method'] = method_list
    df = pd.DataFrame.from_dict(result_dict)
    df = df.melt('method', var_name='Coverage in %', value_name='p_correlation')
    print(df.head())

    #fig = sns.catplot(x='percent', y='p_correlation', hue='method', data=df, kind="box", estimator=np.mean)
    #fig = sns.catplot(x='percent', y='p_correlation', hue='method' ,data=df, kind="point")
    #ax = sns.boxplot(x='Coverage in %', y='p_correlation', hue='method', data=df, dodge=False)
    ax = sns.pointplot(x='Coverage in %', y='p_correlation', hue='method', data=df)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(methods)], labels[:len(methods)]).set_title('Missing Words Methods')
    #fig.set_xlabels('coverage-percent')
    #fig.set_ylabels('p-correlation')
    #fig.set(ylim=(0, 80))
    #fig.ax.set_title('word2vec')
    plt.gca().invert_xaxis()
    plt.show()
    return None

sentences = dev_data['sentence_1'].tolist() + dev_data['sentence_2'].tolist()

test_df = dev_data[abs(dev_data.sentence_len_1 - dev_data.sentence_len_2) < 1]



print(test_df.tail())

pos_percent =get_pos_distribution(sentences)
_, sent_percent = get_sentence_size_distribution(sentences)

methods=["0-vector","hypernym","random","synonym","edit_distance", "jaccard_distance"]

evaluate_rnd_coverage_emb(emb_of_choice, dev_data, 10, percent_mode=True, methods=methods,edit_distance_dic=edit_dist_of_choice, jaccard_distance_dic=jac_dist_of_choice)


test_pos_list= ['NN', 'VB', 'JJ','CD']

pos_filter_similarity(emb=emb_of_choice, data_frame=dev_data,pos_filter=test_pos_list, methods=methods, edit_distance_dic=edit_dist_of_choice, jaccard_distance_dic=jac_dist_of_choice)

missing_words_similarity(emb=emb_of_choice,data_frame=dev_data, edit_distance_dic=edit_dist_of_choice, jaccard_distance_dic=jac_dist_of_choice, methods=methods)
#emb_list = [fastText_emb,google_wv,glove_wv]

#evaluation_emb(emb_list, dev_data)