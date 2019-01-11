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
import missing_words_methods as mwm
import pandas as pd
import embedding as emb
import evaluation as ev
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os


java_path = "C:\\Program Files (x86)\\Common Files\\Oracle\\Java\\javapath\\java.exe"
os.environ['JAVAHOME'] = java_path

st_pos_tagger = StanfordPOSTagger('english-left3words-distsim.tagger', 'C:\\Users\\Rene\\Documents\\stanford-postagger-2018-10-16\\stanford-postagger.jar')

warnings.filterwarnings("ignore")
sentences = []

stemmer = SnowballStemmer("english")
punctation = list(string.punctuation)
stopWords = set(stopwords.words('english'))

#fastText_emb = FastText.load_fasttext_format('Word Embeddings\\wiki.en.bin', 'utf-8')

#fastText_emb = emb.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_fasttext.txt')
#google_wv = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
google_wv = emb.get_word_embeddings('Word Embeddings\\train_embeddings_googleNews_word2vec.txt')
#google_wv_g = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
#glove_wv = emb.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_glove.txt')

#------------------------------
#google_vw_edit_distance = emb.load_words_distance_dict('string_distance_mapping\\train_edit_distance_word2vec.txt')
#google_vw_jaccard_distance = emb.load_words_distance_dict('string_distance_mapping\\train_jaccard_distance_word2vec.txt')

google_vw_edit_distance_all_words = emb.load_words_distance_dict('string_distance_mapping\\all_words_train_edit_distance_word2vec.txt')

emb_of_choice=google_wv
dist_of_choice= google_vw_edit_distance_all_words
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

def get_random_percentage_lists(vec_list_A, vec_list_B, size):
    if not vec_list_A or not vec_list_B:
        return [], []
    list_A_shuffle = vec_list_A.copy()
    random.shuffle(list_A_shuffle)

    list_B_shuffle = vec_list_B.copy()
    random.shuffle(list_B_shuffle)

    # liste mit listen, um später mit all_lists[0] auf list_A_shuffle zuzugreifen und mit all_lists[1] auf list_B_shuffle
    all_lists = [list_A_shuffle, list_B_shuffle]

    # 0.0 - 1.0
    random_percent = size

    # anzahl aller elemente von list_A_new und list_B_new zusammen (bsp 20% bzw 0.2 -> len(list_A) + len(list_B) = 10; 10*0.2 -> 2; also 2 Elemente)
    len_lists_total = int((len(vec_list_A) + len(vec_list_B)) * random_percent)

    # liste mit listen, um später mit all_lists_new[0] auf list_A_new zuzugreifen und mit all_lists_new[1] auf list_B_new
    all_lists_new = [[], []]

    # für die anazahl aller Elemente, die wir hinzufügen:
    for _ in range(len_lists_total):
        # entweder 0 oder 1 -> all_lists[0] = list_A_shuffle und all_lists_new[0] = list_A_new
        random_list_index = random.getrandbits(1)
        # entweder list_A_shuffle oder list_B_shuffle
        random_list = all_lists[random_list_index]
        # wenn die liste leer ist, wir also alle elemente schon in die neue liste gepackt haben, die andere liste benutzen
        if len(random_list) == 0:
        # toggle index: 1 wird 0, 0 wird 1
            random_list_index = 1 - random_list_index
        # also wird die andere liste gewählt
            random_list = all_lists[random_list_index]
        # letztes element entfernen, wegen shuffle ist es ein zufälliger wert
        random_value = random_list.pop()
        # das entfernte element in neue liste packen. Wegen selben index kommen werte aus list_A_shuffle in list_A_new und werte aus list_B_shuffle in list_B_new
        all_lists_new[random_list_index].append(random_value + (True,))

    list_A_new, list_B_new = all_lists_new
    
    if not list_A_new:
        list_A_new.append(all_lists[0].pop() + (True,))
    if not list_B_new:
        list_B_new.append(all_lists[1].pop() + (True,))
    
    while(len(list_A_new) != len(vec_list_A)):
        list_A_new.append(all_lists[0].pop() + (False,))
    while(len(list_B_new) != len(vec_list_B)):
        list_B_new.append(all_lists[1].pop() + (False,))
    return list_A_new, list_B_new


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


def emb_vector_collection(sentence, emb, pos_filter=None):
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
                if pos_filter in pos:
#                    sentence_vec.append(np.zeros(300, dtype="float32"))
                    count_pos+=1
            else:
                missing_token.append(word)
    else:
        for word, pos, use in sentence:
            if word in emb and use:
                sentence_vec.append(emb[word])
                count_pos+=1
                count_use+=1
            if use and word not in emb:
                #sentence_vec.append(mwm.close_distance_word(word=word,distance_dic=dist_of_choice ,embeddings=emb))
                count_use+=1
                sentence_vec.append(synonyme_vec(word=word,pos=pos,emb=emb))
                #sentence_vec.append(np.zeros(300, dtype="float32"))
            if not use:
#                sentence_vec.append(mwm.close_distance_word(word=word,distance_dic=dist_of_choice ,embeddings=emb))
 #               sentence_vec.append(np.zeros(300, dtype="float32"))
               sentence_vec.append(synonyme_vec(word=word,pos=pos,emb=emb))
 #               missing_token.append(word)
 #               count_use+=1
#                sentence_vec.append(synonyme_vec(word=word,pos=pos,emb=emb))
                #sentence_vec.append(np.mean(sentence_emp_vec,axis=0))

    return count_pos,sentence_vec, missing_token, count_use


def rnd_sentence_similarity(emb,sentence_1,sentence_2, size, percent_mode=False, pos_filter=None):
    new_sent1, new_sent2 = get_random_percentage_lists(sentence_1, sentence_2, size)
    count_filter_pos_s1, v2, _, s2_use_count = emb_vector_collection(new_sent2, emb, pos_filter)
    count_filter_pos_s2, v1, _,s1_use_count = emb_vector_collection(new_sent1, emb, pos_filter)
    
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

test_pos_list= ['NN', 'VB', 'JJ','CD']

#print(fmri)
def pos_filter_similarity(data_frame,emb,pos_list):
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
    
def missing_words_similarity(data_frame,emb):
    missing_sentence_count=set((data_frame['missing_emb'].tolist()))
    aktual_miss=[]
    pearson_score=[]
    for word_miss in missing_sentence_count:
        miss_data = data_frame.query('missing_emb =='+str(word_miss))
        miss_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=miss_data.apply(
            lambda row: pd.Series(rnd_sentence_similarity(emb,row['sentence_1'],row['sentence_2'], 1, percent_mode=True)), axis=1)
        p_correlation, p_value = pearsonr(miss_data['pred_value'].tolist(), miss_data['gold_value'].tolist())
        avrg_cov = np.mean(miss_data[['cov_avrg_s1','cov_avrg_s2']].values)
        if len(miss_data.index) > 1:
            pearson_score.append(p_correlation*100)
            aktual_miss.append(word_miss)
        print('--------------- ' + str(word_miss))
        print('Pearson - Correlation: '+ str(p_correlation*100))
        print('Sentence Count: '+ str(len(miss_data.index)))
        print('Avr_Coverage: '+ str(avrg_cov))
    df = pd.DataFrame({'Pearson_Score': pearson_score, 'Missing_Words':aktual_miss})
    sns.pointplot(y='Pearson_Score', x='Missing_Words', data=df).set_title('word2vec')
    plt.show()




# dev_data[['pred_value', 'cov_avrg_s1', 'cov_avrg_s2']]=dev_data.apply(lambda row: pd.Series(rnd_sentence_similarity(emb_of_choice,row['sentence_1'],row['sentence_2'], 1, percent_mode=True)), axis=1)

# p_correlation, p_value = pearsonr(dev_data['pred_value'].tolist(), dev_data['gold_value'].tolist())
# print(p_correlation)

#ax = sns.regplot(dev_data['pred_value'],dev_data['gold_value'], scatter_kws={"color": "black", 'alpha':0.2}, line_kws={"color": "red"})
#ax.set(ylim=(-0.5, 5.5))
#plt.show()
# avrg_cov = np.mean(dev_data[['cov_avrg_s1','cov_avrg_s2']].values)

#avrg_pos_cov_s1 = sum(dev_data['pos_cov_avr_s1'].tolist())/len(dev_data['pos_cov_avr_s1'].tolist())
#avrg_pos_cov_s2 = sum(dev_data['pos_cov_avr_s2'].tolist())/len(dev_data['pos_cov_avr_s2'].tolist())

# print(avrg_cov)
#print((avrg_pos_cov_s1 + avrg_pos_cov_s2)/2)


def evaluate_rnd_coverage_emb(emb, data_frame, iterations, percent_mode):
    if percent_mode is True:
        size=1
    else:
        size =1
    _, axes = plt.subplots(5, 2)
    axes = axes.flatten()
    result_dict={}
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

        result_dict[round(size*100,0)]= result_list
        try:
            ax.set_title(str(round(avr_cov*100,0)))
            sns.distplot(
                result_list,norm_hist=True,ax=ax,label=size)
            

        except np.linalg.linalg.LinAlgError:
             print(result_list) 
        if percent_mode is True:
            size=round(size-0.1,1)
        else:
            size+=1
    plt.show()
    df = pd.DataFrame.from_dict(result_dict)
    print(df.head())
    fig = sns.catplot(data=df, kind="violin", estimator=np.mean)
    sns.catplot(data=df, kind="point", ax=fig.ax, color="black")
    fig.set_xlabels('coverage-percent')
    fig.set_ylabels('p-correlation')
    fig.set(ylim=(0, 80))
    fig.ax.set_title('word2vec')
    plt.show()
    return None

sentences = dev_data['sentence_1'].tolist() + dev_data['sentence_2'].tolist()

test_df = dev_data[abs(dev_data.sentence_len_1 - dev_data.sentence_len_2) < 1]



print(test_df.tail())

pos_percent =get_pos_distribution(sentences)
_, sent_percent = get_sentence_size_distribution(sentences)

evaluate_rnd_coverage_emb(emb_of_choice, dev_data, 100, percent_mode=True)

#pos_filter_similarity(emb=fastText_emb, data_frame=dev_data,pos_list=test_pos_list)

#missing_words_similarity(emb=emb_of_choice,data_frame=dev_data)