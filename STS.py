import csv
import numpy as np
from scipy import spatial
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import pandas as pd
import embedding as emb
import evaluation as ev
import warnings
warnings.filterwarnings("ignore")
sentences = []


def get_word_embeddings(path):
    temp_embeddings={}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split()
            if len(line)>2:
                temp_embeddings[line[0]] = np.asarray(line[1:], dtype=np.float32)
    return temp_embeddings

#word_embed= get_word_embeddings('Word Embeddings\GoogleNews-vectors-negative300.txt')
word_embed={'hallo':2, 'A':3}
def get_word_embeddings_for_sentence(target_path, sentence, emb):
    count=0
    with open(target_path,'w',encoding='utf-8') as f:
        word_list=[]
        for word in sentence:
            if word not in word_list:
                try: 
                    f.write(word + ' ' + emb[word] + '\n')
                except KeyError:
                    count+=1
    print(count)

dev_data =pd.read_csv(
    filepath_or_buffer='stsbenchmark\sts-dev.csv',
#    filepath_or_buffer='./test.csv',
    quoting=csv.QUOTE_NONE,
    sep='\t',
    encoding='utf8',
    header=None,
    usecols=[4,5,6]
    )
dev_data.columns = ['gold_value','sentence_1','sentence_2']

dev_data['sentence_1']=dev_data.apply(lambda row: word_tokenize(row['sentence_1'],'english'), axis=1)
dev_data.apply(lambda row: get_word_embeddings_for_sentence('C:\\Users\\Rene\\Documents\\Semantic Textual Similarity\\Word Embeddings\\new_embddings.txt',row['sentence_1'],word_embed), axis=1)
dev_data['sentence_2']=dev_data.apply(lambda row: word_tokenize(row['sentence_2'],'english'), axis=1)

print(dev_data.tail())


google_wv = KeyedVectors.load_word2vec_format('C:\\Users\\Rene\\Documents\\Semantic Textual Similarity\\Word Embeddings\\GoogleNews-vectors-negative300.txt', binary=False, limit=1000)

#sentence_1 = ['hello', 'my', 'name', 'is', 'obama']
#sentence_2 = ['hello', 'obama', 'my','name', 'is', 'power']
#print(google_wv.n_similarity(sentence_1, sentence_2))
embeddings_words = {}

def sentence_similarity(emb, sentence_1,sentence_2):
    try:
        return emb.n_similarity(sentence_1,sentence_2)
    except KeyError as error:
        print(error)
        return 0
    

dev_data['pred_value']=dev_data.apply(lambda row: sentence_similarity(google_wv,row['sentence_1'],row['sentence_2']), axis=1)

print(dev_data.tail())

#sentence_en_token = [word_tokenize(sentence[0], 'english') for sentence in sentences]
#sentence_es_token = [word_tokenize(sentence[1], 'spanish') for sentence in sentences]


# with open('predict.txt','w') as p:
#     for en, es in zip(sentence_en_token, sentence_es_token):
#         s_vector_en, avr_vec_found_en = emb.sentence_embedding_avg(word_en_embeddings, en ,300)
#         s_vector_es, avr_vec_found_es = emb.sentence_embedding_avg(word_es_embeddings, es , 300)
#         result = 1 - spatial.distance.cosine(s_vector_en, s_vector_es)
#         p.write(str(round(result*5,1))+ '\t'+ str(avr_vec_found_en) + '\t'+ str(avr_vec_found_es) + '\n')
#         print(en, ' ' , es , ': ' ,result)

#print(ev.evaluationB('predict.txt','STS2017.gs/STS.gs.track4a.es-en.txt'))