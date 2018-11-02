
import io
import numpy as np
from scipy import spatial
from nltk.tokenize import word_tokenize
import embedding as emb
sentences = []



embeddings_words = {}
with open('STS2017.eval.v1.1/STS.input.track4a.es-en.txt') as f, open('STS2017.gs/STS.gs.track4a.es-en.txt') as g:
    for sentence_pair, gold_value in zip(f.read().splitlines(), g.read().splitlines()):
        sentence_es, sentence_en = sentence_pair.split('\t')
        sentences.append((sentence_en,sentence_es, gold_value))

sentence_token = [word_tokenize(sentence[0]) for sentence in sentences]

word_embeddings = emb.get_word_embeddings('MWE/english_new.txt')

s_vector_A = emb.sentence_embedding_avg(word_embeddings,sentence_token[0],300)
s_vector_B = emb.sentence_embedding_min(word_embeddings,sentence_token[1])

result = 1 - spatial.distance.cosine(s_vector_A, s_vector_B)
print(sentence_token[0])
print(sentence_token[1])
print(result)