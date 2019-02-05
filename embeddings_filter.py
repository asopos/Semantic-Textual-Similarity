import csv
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from nltk.corpus import wordnet, stopwords
import string
from collections import Counter
import embedding_helper as em

punctation = list(string.punctuation)
stopWords = set(stopwords.words('english'))

data =pd.read_csv(
    filepath_or_buffer='stsbenchmark\\sts-train.csv',
    quoting=csv.QUOTE_NONE,
    sep='\t',
    encoding='utf8',
    header=None,
    usecols=[4,5,6]
    )
data.columns = ['gold_value','sentence_1','sentence_2']
def preprocess_pipeline(sentence):
    word_token = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stopWords]
    word_token = [word for word in word_token if word not in punctation]
    return word_token

data[['sentence_1', 'sentence_2']]=data.apply(
    lambda row: pd.Series([preprocess_pipeline(row['sentence_1']),preprocess_pipeline(row['sentence_2'])]), axis=1)

def save_relevant_embeddings(src_emb, target_path, sen_Token):
    word_list=[]
    count=0
    for sentence in sen_Token:
        for word in sentence:
            word = word.lower()
            if word not in word_list:
                word_list.append(word)
    with open(target_path,'w',encoding='utf-8') as f:
        for word in word_list:
            if word in src_emb:
                f.write(str(word)+ ' '+" ".join(map(str, src_emb[word])) + "\n")
            else:
                count+=1
    print(count)

def save_relevant_embeddings_fastText(src_emb, target_path, sen_Token):
    word_list=[]
    count=0
    for sentence in sen_Token:
        for word in sentence:
            word = word.lower()
            if word not in word_list:
                word_list.append(word)
    with open(target_path,'w',encoding='utf-8') as f:
        for word in word_list:
            if word in src_emb.wv.vocab:
                try: 
                    f.write(str(word)+ ' '+" ".join(map(str, src_emb[word])) + "\n")
                except Exception:
                    print(word)
            else:
                count+=1
    print(count)

def get_missing_words(emb, word_list, target_path):
    with open(target_path,'w',encoding='utf-8') as f:
        for word in word_list:
            if word not in emb:
                f.write(word+ "\n")

def synonym_coverage(word_list):
    syn_count=0
    for word in word_list:
        synset_list = wordnet.synsets(word)
        if synset_list:
            syn_count+=1
        if not synset_list:
            print(word)
    print(syn_count/len(word_list))

def safe_fasttext_txt(fasttext, target_path):
    with open(target_path,'w',encoding='utf-8') as f:
        for word in fasttext.wv.vocab:
            try:
                f.write(str(word)+ ' '+" ".join(map(str, fasttext[word])) + "\n")
            except Exception:
                print(word)

def unique_words(sentence_list):
    word_list=[]
    for sentence in sentence_list:
            for word in sentence:
                if word not in word_list:
                    word_list.append(word)
    return word_list

def sentences_to_words(sentence_list):
    word_list=[]
    for sentence in sentence_list:
        for word in sentence:
            word_list.append(word)
    return word_list

def missing_words_per_sentence(sentence, emb):
    missing_words=[]
    for word in sentence:
        if word not in emb:
            missing_words.append(word)
    return len(missing_words)

def save_distance_word_mapping(word_list,embeddings, target_path):
    print(len(word_list))
    with open(target_path,'w',encoding='utf-8') as f:
        for word in word_list:
            distance_words = [(emb_word, nltk.jaccard_distance(set(emb_word),set(word))) for emb_word in embeddings if emb_word != word]
            new_word = min(distance_words, key = lambda t: t[1])
            f.write(str(word)+ ' '+str(new_word[0]) +' '+ str(new_word[1]) +"\n")


#dev_emb = get_word_embeddings('Word Embeddings\\GoogleNews-vectors-fast-test.txt')
#fastT = em.get_word_embeddings("Word Embeddings\\fasttext_wiki.en.txt")

#google_emb = em.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')

#fastText_emb = em.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_fasttext_word.txt')
#glove_emb = em.get_word_embeddings('Word Embeddings\\glove.6B.300d.txt')



def test_embeddings(emb, sentences):
    words_without_embbedings=[]
    word_list=sentences_to_words(sentences)
    for word in word_list:
        if word.lower() in emb:
            words_without_embbedings.append(word)
    return word_list,words_without_embbedings

sentences =data['sentence_1'].tolist() +  data['sentence_2'].tolist()

u_word_list=unique_words(sentences)
#print(len(test_embeddings(dev_emb,sentences)),len(test_embeddings(google_emb,sentences)))
print('load ready')
#save_relevant_embeddings_fastText(fastT, 'Word Embeddings\\train_embeddings_wiki_glove_official.txt', sentences)
#safe_fasttext_txt(fastT, 'Word Embeddings\\fasttext_wiki.en.txt')
#get_missing_words(fastT,u_word_list, 'missing_words_fasttext_train')
synonym_coverage(u_word_list)

