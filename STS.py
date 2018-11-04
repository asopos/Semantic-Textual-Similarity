
import io
import numpy as np
from scipy import spatial
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import embedding as emb
sentences = []

embeddings_words = {}
with open('STS2017.eval.v1.1/STS.input.track4a.es-en.txt') as f, open('STS2017.gs/STS.gs.track4a.es-en.txt') as g:
    for sentence_pair, gold_value in zip(f.read().splitlines(), g.read().splitlines()):
        sentence_es, sentence_en = sentence_pair.split('\t')
        sentences.append((sentence_en,sentence_es, gold_value))

sentence_en_token = [word_tokenize(sentence[0], 'english') for sentence in sentences]
sentence_es_token = [word_tokenize(sentence[1], 'spanish') for sentence in sentences]

word_en_embeddings = emb.get_word_embeddings('MWE/english_new.txt')
word_es_embeddings = emb.get_word_embeddings('MWE/spanish_new2.txt')

with open('predict.txt','w') as p:
    for en, es in zip(sentence_en_token, sentence_es_token):
        s_vector_en = emb.sentence_embedding_avg(word_en_embeddings, en ,300)
        s_vector_es = emb.sentence_embedding_avg(word_es_embeddings, es , 300)
        result = 1 - spatial.distance.cosine(s_vector_en, s_vector_es)
        p.write(str(result) + '\n')
        # print(en, ' ' , es , ': ' ,result)