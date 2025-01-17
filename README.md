# Semantic-Textual-Similarity

## Vorraussetzungen
- Python 3.7.1
- Installation der Pythonmodule: pip install -r requirements.txt
- Installation der NLTK Module: python -m nltk.downloader all
(Fall die Instalation der NLTK Moduele nich funktioniert: 
  python -> import nltk -> nltk.download()
)
- Herrunterladen der Word Embeddings:
word2vec -> GoogleNews-vectors-negative300.bin.gz: https://code.google.com/archive/p/word2vec/ 
glove -> glove.6B.zip: https://nlp.stanford.edu/projects/glove/#discuss 
fastText -> English: bin+text: https://fasttext.cc/docs/en/pretrained-vectors.html

Die fastText Word Embeddings müssen zunächst aus der .bin Datei in eine .txt Datei umgewandelt werden. 
Dafür ist die Methode safe_fasttext_txt aus embeddings_filter.py vorgesehen.
## Anwendung
Zur Anwendung ist die evaluation.py Datei vorgesehen.

Ein Beispiel zur Ausführung zu word2vec:

Laden der Komponenten
word2vec_wv_g sind Word Embeddings Reduzierung.
word2vec_wv beinhalten ausschließlich Word Embeddings, die im Datensatz vorhanden sind.

word2vec_vw_edit_distance und word2vec_vw_jac_distance sind die Projektionen durch die Abbildungen.

Die Methoden für alternative Wortrepräsentationen sind wie folgt:
methods=["Zufallsvektor","Nullvektor","Hyperonym","Synonym","Levenshtein-Distanz","Jaccard-Distanz"]


Die in der Arbeit vorgestellten Modelle sind in folgende Methoden unterteilt:

### Modell fehlen nach Wörtern
missing_words_similarity(emb=emb_of_choice,data_frame=sts_data, edit_distance_dic=edit_dist_of_choice, 
jaccard_distance_dic=jac_dist_of_choice, methods=methods)

### Randomisierte Wortabdeckung
evaluate_rnd_coverage_emb(emb=emb_of_choice, data_frame=sts_data, iterations=10, percent_mode=True, methods=methods,edit_distance_dic=edit_dist_of_choice, 
jaccard_distance_dic=jac_dist_of_choice,correlationMethod="S")

Für die correlationMethod steht "S" für Spearman und "P" für Pearson

### Filtern nach Wortart
pos_filter_similarity(emb=emb_of_choice, data_frame=sts_data, methods=methods, edit_distance_dic=edit_dist_of_choice,
jaccard_distance_dic=jac_dist_of_choice)


Die Datei embeddings_filter.py beinhaltet Methoden zum Erstellen des Distanz Mappings und Filter um relevante Word Embedings zu extrahieren.

Die reduzierten Word Embeddings und das Mapping der Wortdistanzen ist schon vorhanden.
