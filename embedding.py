import numpy as np

def get_word_embeddings(path):
    temp_embeddings={}
    with open(path) as f:
        for line in f:
            line = line.split()
            temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
    return temp_embeddings

def save_relevant_embeddings(source_path, target_path, sen_Token):
    with open(source_path) as s, open(target_path,'w') as t:
        for line in s:
            word = line.split()
            for tokens in sen_Token:
                if word[0] in tokens:
                    t.write(line)
                    break

def get_sentence_embedding(embeddings, tokens):
    sentence_vector = np.zeros(300, dtype="float32")
    word_count = 0
    for word in embeddings:
        if word in tokens:
            word_count+=1
            sentence_vector = np.add(sentence_vector, embeddings[word])
        
    if word_count>0:
        sentence_vector = np.divide(sentence_vector, word_count)
    return sentence_vector