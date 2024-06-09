# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib  # 用於加載模型
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# def preprocess_text(text):
#     # Download necessary NLTK resources
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('wordnet', quiet=True)
    
#     # Clean the text
#     text = re.sub(r'\W+', ' ', text).lower()
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
#     return ' '.join(tokens)

# # 加載特徵矩陣和標籤數據
# training_data = pd.read_csv('dataset/processed_trainingData.csv')
# test_data = pd.read_csv('dataset/processed_testData.csv')

# # 加載詞頻特徵矩陣
# with np.load('data_processing/training_data/CountVectorizerFeature_model_response.npz', allow_pickle=True) as data:
#     X_train_bow = data['features_matrix'].item()

# # 加載詞頻向量化器
# vectorizer = joblib.load('data_processing/training_data/CountVectorizerVectorizer_model_response.pkl')

# # 預處理測試數據
# test_data['Processed_Resume_str'] = test_data['Resume_str'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else '')

# # 使用訓練時的向量化器處理測試數據
# X_test_bow = vectorizer.transform(test_data['Processed_Resume_str'])

# # 合併公司和職位標籤
# training_data['label'] = training_data['company_name'] + '_' + training_data['position_title']
# y_train = training_data['label']

# # 拆分訓練和驗證數據
# X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_bow, y_train, test_size=0.2, random_state=42)

# # 訓練模型（以邏輯迴歸為例）
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train_part, y_train_part)

# # 驗證模型
# y_val_pred = model.predict(X_val)
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print(f"Validation Accuracy: {val_accuracy}")

# # 對測試數據進行預測
# test_predictions = model.predict(X_test_bow)

# # 保存預測結果
# test_data['Predicted_Label'] = test_predictions
# test_data.to_csv('find_job/test_data_with_predictions.csv', index=False)


# print(training_data.info())
# print(training_data.head())
# print(training_data['label'].value_counts())

# print(X_train_bow.shape)
# print(X_test_bow.shape)

# print(training_data['label'].value_counts())



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # 用於加載模型
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

# 加載特徵矩陣和標籤數據
training_data = pd.read_csv('dataset/processed_trainingData.csv')
test_data = pd.read_csv('dataset/processed_testData.csv')

# 加載 TF-IDF 特徵矩陣
with np.load('data_processing/training_data/TfidfVectorizerFeature_model_response.npz', allow_pickle=True) as data:
    X_train_tfidf = data['features_matrix'].item()

# 加載 TF-IDF 向量化器
vectorizer = joblib.load('data_processing/training_data/TfidfVectorizerVectorizer_model_response.pkl')

# 預處理測試數據
test_data['Processed_Resume_str'] = test_data['Resume_str'].apply(lambda x: preprocess_text(x) if pd.notnull(x) else '')

# 使用訓練時的向量化器處理測試數據
X_test_tfidf = vectorizer.transform(test_data['Processed_Resume_str'])

# 合併公司和職位標籤
training_data['label'] = training_data['company_name'] + '_' + training_data['position_title']
y_train = training_data['label']

# 拆分訓練和驗證數據
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# 訓練模型（以隨機森林為例）
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_part, y_train_part)

# 驗證模型
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# 對測試數據進行預測
test_predictions = model.predict(X_test_tfidf)

# 保存預測結果
test_data['Predicted_Label'] = test_predictions
test_data.to_csv('find_job/test_data_with_predictions.csv', index=False)
