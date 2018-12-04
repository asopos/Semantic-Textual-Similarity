import numpy as np
from nltk.tag import pos_tag

def get_word_embeddings(path):
    temp_embeddings={}
    with open(path) as f:
        for line in f:
            line = line.split()
            try:
                temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
            except ValueError:
                temp_embeddings[line[1]] = np.asarray(line[2:], dtype=np.float32)
    return temp_embeddings

def save_relevant_embeddings(source_path, target_path, sen_Token):
    with open(source_path, encoding='utf-8') as s, open(target_path,'w', encoding='utf-8') as t:
        for line in s:
            word = line.split()
            for tokens in sen_Token:
                if word[0] in tokens:
                    t.write(line)
                    break

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