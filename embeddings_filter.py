import csv
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from nltk.corpus import wordnet, stopwords
import string
from collections import Counter

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

def get_word_embeddings(path):
    temp_embeddings={}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split()
            if len(line)>2:
                try:
                    temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
                except ValueError:
                    print(line[0])
    return temp_embeddings

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
            try:
                f.write(str(word)+ ' '+" ".join(map(str, src_emb[word])) + "\n")
            except KeyError:
                count+=1
    print(count)

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
#    distance_words = map(lambda x:(x[0],nltk.edit_distance(x,word)), embeddings)
    print(len(word_list))
    with open(target_path,'w',encoding='utf-8') as f:
        for word in word_list:
            distance_words = [(emb_word, nltk.edit_distance(emb_word,word)) for emb_word in embeddings if emb_word != word]
            new_word = min(distance_words, key = lambda t: t[1])
            f.write(str(word)+ ' '+str(new_word[0]) +' '+ str(new_word[1]) +"\n")


#dev_emb = get_word_embeddings('Word Embeddings\\GoogleNews-vectors-fast-test.txt')
google_emb = get_word_embeddings('Word Embeddings\\train_embeddings_googleNews_word2vec.txt')
#fastText_emb = FastText.load_fasttext_format('Word Embeddings\\wiki.en.bin', 'utf-8')
#glove_emb = get_word_embeddings('Word Embeddings\\glove.txt')

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

#save_relevant_embeddings(glove_emb, 'Word Embeddings\\train_embeddings_wiki_glove.txt', sentences)


save_distance_word_mapping(word_list=u_word_list, embeddings=google_emb, target_path='string_distance_mapping\\test_train_edit_distance_word2vec.txt')