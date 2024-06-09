import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_features(file_path):
    with np.load(file_path, allow_pickle=True) as data:
        features = data['features_matrix'].item()
    return features

def interpret_clusters(features, labels, num_clusters, feature_names, top_n=10):
    cluster_centers = []
    cluster_keywords = []
    for i in range(num_clusters):
        cluster_center = features[labels == i].mean(axis=0)
        cluster_centers.append(cluster_center)

        # 找出每个簇中的关键字
        cluster_indices = np.where(labels == i)[0]
        cluster_features = features[cluster_indices]
        keyword_counts = np.sum(cluster_features, axis=0).A1  # 确保 keyword_counts 是一维的
        keyword_indices = np.argsort(keyword_counts)[::-1][:top_n]  # 取前 top_n 个关键字

        # 确保索引在特征名称的范围内，并且只选择前 top_n 个关键字
        keywords = [str(feature_names[index]) for index, count in zip(keyword_indices, keyword_counts[keyword_indices]) if index < len(feature_names) and count > 0]
        cluster_keywords.append(keywords[:top_n])

    cluster_centers = np.array(cluster_centers).reshape(num_clusters, -1)
    return cluster_centers, cluster_keywords

def calculate_similarity(job_description, resumes, vectorizer):
    job_desc_features = vectorizer.transform([job_description])
    resume_features = vectorizer.transform(resumes)
    similarity_scores = cosine_similarity(job_desc_features, resume_features)
    return similarity_scores

def find_most_similar_cluster(job_description, cluster_centers, vectorizer):
    job_desc_features = vectorizer.transform([job_description])
    cluster_similarities = cosine_similarity(job_desc_features, cluster_centers)
    most_similar_cluster_index = np.argmax(cluster_similarities)
    return most_similar_cluster_index, cluster_similarities

def main():
    # 加载预处理生成的特征文件
    tfidf_features_path = 'data_processing/test_data/TfidfVectorizerFeature_Resume_str.npz'
    vectorizer_path = 'data_processing/test_data/TfidfVectorizerVectorizer_Resume_str.pkl'
    # job_tfidf_features_path = 'data_processing/training_data/TfidfVectorizerFeature_model_response.npz'
    job_vectorizer_path = 'data_processing/training_data/TfidfVectorizerVectorizer_model_response.pkl'

    # 加载特征
    tfidf_features = load_features(tfidf_features_path)
    # job_tfidf_features = load_features(job_tfidf_features_path)

    # 加载向量化器
    vectorizer = joblib.load(vectorizer_path)
    job_vectorizer = joblib.load(job_vectorizer_path)
    feature_names = vectorizer.get_feature_names_out()
    
    # 打印特征名称数组的长度
    print(f"Feature names length: {len(feature_names)}")

    # 选择要使用的特征进行聚类（这里以 TF-IDF 为例）
    features = tfidf_features

    # 设置K值，假设为5
    num_clusters = 5

    # 使用 K-means 进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)

    # 获取聚类结果
    labels = kmeans.labels_

    # 解释簇中心和关键字
    cluster_centers, cluster_keywords = interpret_clusters(features, labels, num_clusters, feature_names)
    cluster_centers_df = pd.DataFrame(cluster_centers)
    cluster_centers_df.to_csv('clustering_analysis/cluster_centers.csv', index=False)
    print("Cluster centers have been saved to 'cluster_centers.csv'.")
    
    # 打印每个簇的关键字
    for i, keywords in enumerate(cluster_keywords):
        print(f"Cluster {i} keywords: {', '.join(keywords)}")

    ##########

    # 加載職位描述數據
    job_descriptions_df = pd.read_csv('dataset/processed_trainingData.csv')
    # 假設要分析的職位描述是第一個職位描述
    job_description = job_descriptions_df['Processed_model_response'].iloc[10]

    # 找到与职位描述最相似的聚类簇
    most_similar_cluster_index, cluster_similarities = find_most_similar_cluster(job_description, cluster_centers, vectorizer)
    print(f"The most similar cluster to the job description is cluster {most_similar_cluster_index} with similarity scores {cluster_similarities}")
    similar_cluster_indices = np.where(labels == most_similar_cluster_index)[0]
    print(f"Resumes in the most similar cluster: {similar_cluster_indices}")

    ##########
    
    # 使用 PCA 进行降维
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(features.toarray())

    # 绘制聚类结果
    colors = ["r", "b", "c", "y", "m"]
    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]
    plt.figure(figsize=(10, 6))

    # 绘制每个簇的点
    for i in range(num_clusters):
        cluster_points = np.array([scatter_plot_points[j] for j in range(len(labels)) if labels[j] == i])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}')

    plt.title("Clustered Resumes and Job Descriptions")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()