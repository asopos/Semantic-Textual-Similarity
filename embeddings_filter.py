import csv
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
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
                temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
    return temp_embeddings

def get_word_embeddings_for_sentence(source_path,target_path, sentences):
    word_list=[]
    count=0
    for sentence in sentences:
        for word in sentence:
            if word not in word_list:
                word_list.append(word)
    with open(source_path, encoding='utf-8') as s, open(target_path,'w',encoding='utf-8') as f:
        for line in s:
            sline = line.split()
            if sline[0] in word_list:
                f.write(line)
            else:
                count+=1
    print(count)

sentences = dev_data['sentence_1'].tolist() + dev_data['sentence_2'].tolist()

get_word_embeddings_for_sentence('C:\\Users\\Rene\\Documents\\Semantic Textual Similarity\\Word Embeddings\\dev_data_embeddings.txt','C:\\Users\\Rene\\Documents\\Semantic Textual Similarity\\Word Embeddings\\dev_data_embeddings_test.txt',sentences)