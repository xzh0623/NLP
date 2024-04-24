from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
from scipy.stats import spearmanr

# GloVe to Word2Vec format
glove_input_file = 'glove.twitter.27B/glove.twitter.27B.100d.txt'
word2vec_output_file = 'glove.twitter.27B/glove.twitter.27B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

# laod model after transforming
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Load your data (adjust path as necessary)
data_path = 'wordsim353/combined.csv'
data = pd.read_csv(data_path)

# Calculate model similarity and compare with human ratings
word = []
predicted_similarities = []
for index, row in data.iterrows():
    word1 = row['Word 1']
    word2 = row['Word 2']
    # Ensure words are in the model to avoid KeyError
    if word1 in model.key_to_index and word2 in model.key_to_index:
        model_similarity = model.similarity(word1, word2)
        predicted_similarities.append(model_similarity)
        word.append([word1, word2])
    else:
        predicted_similarities.append(None)  # Or handle missing words appropriately

for i in range(len(word)):
    print(str(word[i]) + ":" + str(predicted_similarities[i]))

# Calculate correlation
data['predicted_similarity'] = predicted_similarities
correlation, _ = spearmanr(data.dropna()['Human (mean)'], data.dropna()['predicted_similarity'])
print("Spearman correlation:", correlation)