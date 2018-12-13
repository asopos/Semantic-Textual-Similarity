import csv
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import FastText
dev_data =pd.read_csv(
    filepath_or_buffer='stsbenchmark\\sts-dev.csv',
    quoting=csv.QUOTE_NONE,
    sep='\t',
    encoding='utf8',
    header=None,
    usecols=[4,5,6]
    )
dev_data.columns = ['gold_value','sentence_1','sentence_2']
dev_data['sentence_1']=dev_data.apply(lambda row: word_tokenize(row['sentence_1'],'english'), axis=1)
dev_data['sentence_2']=dev_data.apply(lambda row: word_tokenize(row['sentence_2'],'english'), axis=1)

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



sentences = dev_data['sentence_1'].tolist() + dev_data['sentence_2'].tolist()

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

#dev_emb = get_word_embeddings('Word Embeddings\\GoogleNews-vectors-fast-test.txt')
#google_emb = get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')
fastText_emb = FastText.load_fasttext_format('Word Embeddings\\wiki.en.bin', 'utf-8')
#glove_emb = get_word_embeddings('Word Embeddings\\glove.txt')

def test_embeddings(emb, sentences):
    words_without_embbedings=[]
    word_list=sentences_to_words(sentences)
    for word in word_list:
        try:
            a=emb[word.lower()]
        except KeyError:
            words_without_embbedings.append(word)
    return words_without_embbedings


#print(len(test_embeddings(dev_emb,sentences)),len(test_embeddings(google_emb,sentences)))

save_relevant_embeddings(fastText_emb, 'Word Embeddings\\dev_embeddings_wiki_fasttext.txt', sentences)