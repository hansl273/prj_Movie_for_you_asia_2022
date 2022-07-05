import pandas as pd

df = pd.read_csv('./crawling_data/cleaned_review_2020.csv')
df.dropna(inplace=True)
df.info()
one_sentences = []
for title in df['title'].unique():
    temp = df[df['title'] == title]
    if len(temp) > 30:
        temp = temp.iloc[:30, :]
    one_sentence = ' '.join(temp['cleaned_sentences'])
    one_sentences.append(one_sentence)
df_one = pd.DataFrame({'titles':df['title'].unique(), 'reviews':one_sentences})
print(df_one.head())
df_one.to_csv('./crawling_data/cleaned_review_one.csv', index=False)






