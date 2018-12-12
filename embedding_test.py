from gensim.models import FastText

fastText_emb = FastText.load_fasttext_format('Word Embeddings\\fastText_Skipgram_WIki.bin', 'utf-8')

sent_1 =['Hello','i','AM','obama']

sent_2 = ['I','am', 'the', 'President', 'of', 'the', 'usa']

print(fastText_emb['o'])