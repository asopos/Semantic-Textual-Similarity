
import io
import numpy as np
from nltk.tokenize import word_tokenize
sentences = []


embeddings_words = {}
with open('STS2017.eval.v1.1/STS.input.track4a.es-en.txt') as f, open('STS2017.gs/STS.gs.track4a.es-en.txt') as g:
    for sentence_pair, gold_value in zip(f.read().splitlines(), g.read().splitlines()):
        sentence_es, sentence_en = sentence_pair.split('\t')
        sentences.append((sentence_en,sentence_es, gold_value))

sentence_token = [word_tokenize(sentence[0]) for sentence in sentences]
# with open('/Users/renelehmann/Documents/Semantic-Textual-Similarity/MWE/english.txt') as f:
#     for line in f:
#         line = line.split()
#         for tokens in sentence_token:
#             if line[0] in tokens:
#                 embeddings_words[line[0]] = line[1:]
#                 print(line[0])
#                 break

def write_to_file(path, embeddings):
    with open(path,'w') as f:
        for embedding in embeddings:
            f.write(embedding)
    return True

def save_embeddings(source_path, target_path, sen_Token):
    with open(source_path) as s, open(target_path,'w') as t:
        for line in s:
            word = line.split()
            for tokens in sen_Token:
                if word[0] in tokens:
                    t.write(line)
                    break

save_embeddings(
'/Users/renelehmann/Documents/Semantic-Textual-Similarity/MWE/english.txt',
'/Users/renelehmann/Documents/Semantic-Textual-Similarity/MWE/english_new.txt',
sentence_token)


def avg_sentence_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec