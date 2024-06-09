import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import os
from scipy.sparse import csr_matrix
import joblib  # 用於保存和加載模型

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    # Download necessary NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    # Clean the text
    text = re.sub(r'\W+', ' ', text).lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def apply_preprocessing(data, columns):
    for column in columns:
        data['Processed_' + column] = data[column].apply(lambda x: preprocess_text(x) if pd.notnull(x) else '')
    return data

def save_processed_data(data, file_path):
    data.to_csv(file_path, index=False)

def vectorize_data(data, vectorizer, feature_name, data_type):
    if data_type == 'training':
        features = vectorizer.fit_transform(data['Processed_' + feature_name])
        save_path = f'data_processing/training_data/{vectorizer.__class__.__name__}Feature_{feature_name}.npz'
        model_path = f'data_processing/training_data/{vectorizer.__class__.__name__}Vectorizer_{feature_name}.pkl'
    else:
        features = vectorizer.transform(data['Processed_' + feature_name])
        save_path = f'data_processing/test_data/{vectorizer.__class__.__name__}Feature_{feature_name}.npz'
        model_path = f'data_processing/test_data/{vectorizer.__class__.__name__}Vectorizer_{feature_name}.pkl'

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    # Specify the array name explicitly when saving
    np.savez_compressed(save_path, features_matrix=features)
    joblib.dump(vectorizer, model_path)  # 保存向量化器
    return vectorizer, features

def train_word2vec(data, feature_name, data_type):
    sentences = [row.split() for row in data['Processed_' + feature_name]]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.train(sentences, total_examples=len(sentences), epochs=10)
    
    if data_type == 'training':
        save_path = f'data_processing/training_data/word2vecModel_{feature_name}.w2v'
    else:
        save_path = f'data_processing/test_data/word2vecModel_{feature_name}.w2v'

    model.save(save_path)

def load_and_print_matrix_details(file_path, feature_name):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            print(f"- Contents of {feature_name}:")
            for key in data.files:
                array = data[key]
                if isinstance(array.item(), csr_matrix):
                    matrix = array.item()
                    print(f"{key}: shape = {matrix.shape}")
                    print(f"Contents of {key}:")
                    print(matrix)
                else:
                    print(f"{key}: shape = {array.shape}")
                    print(f"Contents of {key}:")
                    print(array)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {str(e)}")

def print_feature_details():
    print("Training Data:")
    load_and_print_matrix_details('data_processing/training_data/CountVectorizerFeature_model_response.npz', 'modelResponse BoW')
    load_and_print_matrix_details('data_processing/training_data/CountVectorizerFeature_position_title.npz', 'positionTitle BoW')
    load_and_print_matrix_details('data_processing/training_data/TfidfVectorizerFeature_model_response.npz', 'modelResponse TF-IDF')
    load_and_print_matrix_details('data_processing/training_data/TfidfVectorizerFeature_position_title.npz', 'positionTitle TF-IDF')

    print("\nTesting Data:")
    load_and_print_matrix_details('data_processing/test_data/CountVectorizerFeature_Resume_str.npz', 'Resume_str BoW')
    load_and_print_matrix_details('data_processing/test_data/TfidfVectorizerFeature_Resume_str.npz', 'Resume_str TF-IDF')

def load_matrix_and_get_value(file_path, row_index, col_index):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            matrix = data['features_matrix'].item()  # 假設數據存儲在 'features_matrix' 鍵下
            value = matrix[row_index, col_index]
            return value
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except KeyError:
        print("Key error: ensure the correct key is being accessed.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    training_data = load_data('dataset/training_data.csv')
    test_data = load_data('dataset/test_data.csv')
    
    # Preprocess data
    training_columns = ['model_response', 'position_title']
    test_columns = ['Resume_str']
    training_data = apply_preprocessing(training_data, training_columns)
    test_data = apply_preprocessing(test_data, test_columns)
    
    # Save processed data
    save_processed_data(training_data, 'dataset/processed_trainingData.csv')
    save_processed_data(test_data, 'dataset/processed_testData.csv')
    
    # Feature extraction
    vectorizer = CountVectorizer()
    vectorizer, _ = vectorize_data(training_data, vectorizer, 'model_response', 'training')
    vectorizer, _ = vectorize_data(training_data, vectorizer, 'position_title', 'training')
    vectorizer, _ = vectorize_data(test_data, vectorizer, 'Resume_str', 'test')
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer, _ = vectorize_data(training_data, tfidf_vectorizer, 'model_response', 'training')
    tfidf_vectorizer, _ = vectorize_data(training_data, tfidf_vectorizer, 'position_title', 'training')
    tfidf_vectorizer, _ = vectorize_data(test_data, tfidf_vectorizer, 'Resume_str', 'test')
    
    train_word2vec(training_data, 'model_response', 'training')
    train_word2vec(training_data, 'position_title', 'training')
    train_word2vec(test_data, 'Resume_str', 'test')
    
    print_feature_details()
    
    file_path = 'data_processing/training_data/CountVectorizerFeature_model_response.npz'
    row_index = 0
    col_index = 1229
    value = load_matrix_and_get_value(file_path, row_index, col_index)
    print(f"The value at position ({row_index}, {col_index}) is: {value}")


if __name__ == '__main__':
    main()