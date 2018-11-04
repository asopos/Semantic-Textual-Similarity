import numpy as np

def get_word_embeddings(path):
    temp_embeddings={}
    with open(path) as f:
        for line in f:
            line = line.split()
            try:
                temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
            except ValueError:
                word = line[0]+ ' ' + line[1]
                temp_embeddings[word] = np.asarray(line[2:], dtype=np.float32)
    return temp_embeddings

def save_relevant_embeddings(source_path, target_path, sen_Token):
    with open(source_path) as s, open(target_path,'w') as t:
        for line in s:
            word = line.split()
            for tokens in sen_Token:
                if word[0] in tokens:
                    t.write(line)
                    break

def sentence_embedding_avg(word_embeddings, tokens,dimension):
    sentence_vector = np.zeros(dimension, dtype="float32")
    word_count = 0
    for word in word_embeddings:
        if word in tokens:
            word_count+=1
            sentence_vector = np.add(sentence_vector, word_embeddings[word])
        
    if word_count>0:
        sentence_vector = np.divide(sentence_vector, word_count)
    return sentence_vector

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