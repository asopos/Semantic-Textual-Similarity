
import io
import numpy as np
sentences = []


embeddings_words = []
words_en = []
with open('STS2017.eval.v1.1/STS.input.track4a.es-en.txt') as f, open('STS2017.gs/STS.gs.track4a.es-en.txt') as g:
    for sentence_pair, gold_value in zip(f.read().splitlines(), g.read().splitlines()):
        sentence_es, sentence_en = sentence_pair.split('\t')
        sentences.append((sentence_en,sentence_es, gold_value))
    
with open('/Users/renelehmann/Documents/Semantic-Textual-Similarity/MWE/english.txt') as f:
    for line in f:
        line = line.split()
#        if line[0] in sentence_en and len(line[0]) > 2:
        embeddings_words.append(line[0])
for sentence in sentences:
    for word in sentence.split():
        if word in embeddings_words:
            print(word)


print(len(embeddings_words))


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