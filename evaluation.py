import csv

import warnings
import pandas as pd

import embedding_helper as emb
import evaluation_methods as em
import sts_methods as sts

warnings.filterwarnings("ignore")
#------------Word Embeddings

#fastText_emb = emb.get_word_embeddings('Word Embeddings\\all_embeddings_wiki_fasttext.txt')
fastText_emb_g = emb.get_word_embeddings('Word Embeddings\\fasttext_wiki.en.txt')

#word2vec_wv = emb.get_word_embeddings('Word Embeddings\\all_embeddings_googleNews_word2vec.txt')
#word2vec_wv_g = emb.get_word_embeddings('Word Embeddings\\GoogleNews-vectors-negative300.txt')

#glove_wv = emb.get_word_embeddings('Word Embeddings\\train_embeddings_wiki_glove_official.txt')
#glove_wv_g = emb.get_word_embeddings('Word Embeddings\\glove.6B.300d.txt')
#-------------Wort Distanzen

#word2vec_vw_edit_distance = emb.load_words_distance_dict('string_distance_mapping\\all_edit_distance_word2vec.txt')
#word2vec_vw_jac_distance = emb.load_words_distance_dict('string_distance_mapping\\all_jaccard_distance_word2vec.txt')
fasttext_vw_edit_distance = emb.load_words_distance_dict('string_distance_mapping\\all_edit_distance_fasttext.txt')
fasttext_vw_jac_distance = emb.load_words_distance_dict('string_distance_mapping\\all_jaccard_distance_fasttext.txt')
#glove_wv_edit_distance = emb.load_words_distance_dict('string_distance_mapping\\all_edit_distance_glove.txt')
#glove_wv_jac_distance = emb.load_words_distance_dict('string_distance_mapping\\all_jaccard_distance_glove.txt')

methods=["Zufallsvektor","Nullvektor","Hyperonym","Synonym","Levenshtein-Distanz","Jaccard-Distanz"]

emb_of_choice=fastText_emb_g
edit_dist_of_choice= fasttext_vw_edit_distance
jac_dist_of_choice = fasttext_vw_jac_distance

sts_data =pd.read_csv(
    filepath_or_buffer='stsbenchmark\\sts-all.csv',
    quoting=csv.QUOTE_NONE,
    sep='\t',
    encoding='utf8',
    header=None,
    usecols=[4,5,6]
    )
sts_data.columns = ['gold_value','sentence_1','sentence_2']

sts_data[['sentence_1', 'sentence_2']]=sts_data.apply(
    lambda row: pd.Series([sts.preprocess_pipeline(row['sentence_1']),sts.preprocess_pipeline(row['sentence_2'])]), axis=1)

sts_data['missing_emb'] = sts_data.apply(lambda row: sts.missing_words_per_sentence(row['sentence_1']+ row['sentence_2'], emb_of_choice), axis=1)

sts_data[['sentence_len_1', 'sentence_len_2']]=sts_data.apply(
    lambda row: pd.Series([len(row['sentence_1']),len(row['sentence_2'])]), axis=1)


#Keine leeren Sätze nach Stopwordfilter
sts_data = sts_data[(sts_data['sentence_len_1']>0) & (sts_data['sentence_len_2']>0)]

print(sts_data.head())


em.missing_words_similarity(emb=emb_of_choice,data_frame=sts_data, edit_distance_dic=edit_dist_of_choice, jaccard_distance_dic=jac_dist_of_choice, methods=methods)

em.evaluate_rnd_coverage_emb(emb=emb_of_choice, data_frame=sts_data, iterations=10, percent_mode=True, methods=methods,edit_distance_dic=edit_dist_of_choice, jaccard_distance_dic=jac_dist_of_choice,correlationMethod="P")

em.method_evaluation(emb=emb_of_choice,data_frame=sts_data, edit_distance_dic=edit_dist_of_choice, jaccard_distance_dic=jac_dist_of_choice, methods=methods)

em.pos_filter_similarity(emb=emb_of_choice, data_frame=sts_data, methods=methods, edit_distance_dic=edit_dist_of_choice, jaccard_distance_dic=jac_dist_of_choice)