
import io
import numpy as np
from scipy import spatial
from nltk.tokenize import word_tokenize
sentences = []



embeddings_words = {}
with open('STS2017.eval.v1.1/STS.input.track4a.es-en.txt') as f, open('STS2017.gs/STS.gs.track4a.es-en.txt') as g:
    for sentence_pair, gold_value in zip(f.read().splitlines(), g.read().splitlines()):
        sentence_es, sentence_en = sentence_pair.split('\t')
        sentences.append((sentence_en,sentence_es, gold_value))

sentence_token = [word_tokenize(sentence[0]) for sentence in sentences]

def get_word_embeddings(path):
    temp_embeddings={}
    with open(path) as f:
        for line in f:
            line = line.split()
            temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
    return temp_embeddings

def save_embeddings(source_path, target_path, sen_Token):
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

word_embeddings = get_word_embeddings('MWE/english_new.txt')

s_vector_A = get_sentence_embedding(word_embeddings,sentence_token[0])
s_vector_B = get_sentence_embedding(word_embeddings,sentence_token[1])

result = 1 - spatial.distance.cosine(s_vector_A, s_vector_B)
print(sentence_token[0])
print(sentence_token[1])
print(result)