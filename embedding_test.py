relevant_missing_words = []
google_missing_words =[]

diffrences=[]

with open('missing_token.txt', encoding='utf-8') as s:
    for line in s:
        relevant_missing_words.append(str(line))

with open('missing_token_all_G.txt', encoding='utf-8') as s:
    for line in s:
        google_missing_words.append(str(line))

for word in relevant_missing_words:
    if word not in google_missing_words:
        diffrences.append(word)


print(len(diffrences))
print(diffrences)


print(len(relevant_missing_words),len(google_missing_words))