import numpy as np
from nltk.tag import pos_tag
from gensim import matutils

def get_word_embeddings(path):
    temp_embeddings={}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split()
            if len(line)>2:
                temp_embeddings[line[0].lower()] = np.asarray(line[1:], dtype=np.float32)
    return temp_embeddings

def save_relevant_embeddings(source_path, target_path, sen_Token):
    word_list=[]
    count=0
    for sentence in sen_Token:
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

def sentence_embedding_avg(word_embeddings, tokens,dimension):
    sentence_vector = np.zeros(dimension, dtype="float32")
    emb_count = 0
    for word in word_embeddings:
        if word in tokens:
            emb_count+=1
            sentence_vector = np.add(sentence_vector, word_embeddings[word])
    if emb_count>0:
        sentence_vector = np.divide(sentence_vector, emb_count)
    emb_avrg=emb_count/len(tokens)
    return sentence_vector, emb_avrg

def sentence_embedding_POS(word_embeddings, tokens, dimension):
    sentence_vector = np.zeros(dimension, dtype="float32")
    for token in tokens:
        if token in word_embeddings:
            sentence_vector = np.add(sentence_vector, word_embeddings[token])
    return None

def sentence_embedding_min(word_embeddings, tokens):
    temp_word_embeddings= []
    for word in word_embeddings:
        if word in tokens:
            temp_word_embeddings.append(word_embeddings[word])
    return np.asarray(temp_word_embeddings).min(0)

def sentence_embedding_max(word_embeddings, tokens):
    temp_word_embeddings= []
    for word in word_embeddings:
        if word in tokens:
            temp_word_embeddings.append(word_embeddings[word])
    return np.asarray(temp_word_embeddings).max(0)

def sentence_similarity(emb,sentence_1,sentence_2):
    v1 = []
    v2 = []
    try:
        v1 = [emb[word.lower()] for word in sentence_1]
        v2 = [emb[word.lower()] for word in sentence_2]
    except KeyError as error:
        print(error)
    return np.dot(matutils.unitvec(np.array(v1).mean(axis=0)), matutils.unitvec(np.array(v2).mean(axis=0)))