import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# 加載預訓練的 GloVe 詞嵌入
def load_glove_model(glove_file_path):
    glove_model = {}
    with open(glove_file_path, "r", encoding="utf8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    return glove_model

# 將文本轉換為 GloVe 向量
def text_to_glove_vector(text, glove_model, embedding_dim=300):
    if isinstance(text, float):  # 檢查是否為浮點數（如 NaN）
        return np.zeros(embedding_dim)
    words = text.split()
    word_vectors = [glove_model.get(word, np.zeros(embedding_dim)) for word in words]
    if len(word_vectors) == 0:
        return np.zeros(embedding_dim)
    return np.mean(word_vectors, axis=0)

# 加載數據並轉換為 GloVe 向量
def load_and_transform_data(file_path, glove_model, text_column):
    data = pd.read_csv(file_path)
    data['glove_vector'] = data[text_column].apply(lambda x: text_to_glove_vector(x, glove_model))
    return data

def interpret_clusters(features, labels, num_clusters, resume_data, text_column, top_n=10):
    cluster_centers = []
    cluster_keywords = []
    for i in range(num_clusters):
        cluster_center = features[labels == i].mean(axis=0)
        cluster_centers.append(cluster_center)

        # 找出每個簇中的關鍵字
        cluster_indices = np.where(labels == i)[0]
        cluster_resumes = resume_data.iloc[cluster_indices][text_column]
        keyword_counts = {}
        
        for resume in cluster_resumes:
            if isinstance(resume, float):  # 檢查是否為浮點數（如 NaN）
                continue
            words = resume.split()
            for word in words:
                if word in keyword_counts:
                    keyword_counts[word] += 1
                else:
                    keyword_counts[word] = 1

        # 取前 top_n 個關鍵字
        sorted_keywords = sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)
        keywords = [keyword for keyword, count in sorted_keywords[:top_n]]
        cluster_keywords.append(keywords)

    cluster_centers = np.array(cluster_centers).reshape(num_clusters, -1)
    return cluster_centers, cluster_keywords

# 計算職位描述和履歷之間的相似度
def calculate_similarity(job_description, resumes, glove_model):
    job_desc_vector = text_to_glove_vector(job_description, glove_model)
    resume_vectors = np.array(resumes['glove_vector'].tolist())
    similarity_scores = cosine_similarity([job_desc_vector], resume_vectors)
    return similarity_scores

def find_most_similar_cluster(job_description, cluster_centers, glove_model):
    job_desc_features = text_to_glove_vector(job_description, glove_model)
    cluster_similarities = cosine_similarity([job_desc_features], cluster_centers)
    most_similar_cluster_index = np.argmax(cluster_similarities)
    return most_similar_cluster_index, cluster_similarities

# 保存排序結果到CSV文件
def save_sorted_results_to_csv(results, output_file):
    results.to_csv(output_file, index=False)

def main():
    # GloVe 文件路徑
    glove_file_path = '/Users/chantsaiching/Desktop/glove.6B/glove.6B.300d.txt'
    
    # 加載 GloVe 模型
    glove_model = load_glove_model(glove_file_path)
    
    # 加載並轉換數據
    resume_data_path = 'dataset/processed_testData.csv'
    job_data_path = 'dataset/processed_trainingData.csv'
    
    resume_data = load_and_transform_data(resume_data_path, glove_model, 'Processed_Resume_str')
    job_data = load_and_transform_data(job_data_path, glove_model, 'Processed_model_response')
    
    # 進行聚類
    features = np.array(resume_data['glove_vector'].tolist())
    num_clusters = 5
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    
    labels = kmeans.labels_
    
    # 解析簇中心和關鍵詞
    cluster_centers, cluster_keywords = interpret_clusters(features, labels, num_clusters, resume_data, 'Processed_Resume_str', top_n=10)
    cluster_centers_df = pd.DataFrame(cluster_centers)
    cluster_centers_df.to_csv('clustering_analysis/cluster_centers.csv', index=False)
    print("Cluster centers have been saved to 'cluster_centers.csv'.")
    
    for i, keywords in enumerate(cluster_keywords):
        print(f"Cluster {i} keywords: {', '.join(keywords)}")
    
    # 保存所有結果的DataFrame
    all_results = []

    for job_index in range(len(job_data)):
        # 加載職位描述數據並計算相似度
        job_description = job_data.iloc[job_index]['Processed_model_response']
        company_name = job_data.iloc[job_index]['company_name']
        position_title = job_data.iloc[job_index]['position_title']
        # print(f"Processing job {job_index + 1}/{len(job_data)}: {company_name}, Position Title: {position_title}")

        most_similar_cluster_index, cluster_similarities = find_most_similar_cluster(job_description, cluster_centers, glove_model)
        # print(f"The most similar cluster to the job description is cluster {most_similar_cluster_index} with similarity scores {cluster_similarities}")
        
        # 找到最相似簇中的履歷 ID 並排序
        similar_cluster_indices = np.where(labels == most_similar_cluster_index)[0]
        similar_resumes = resume_data.iloc[similar_cluster_indices]
        similarity_scores = calculate_similarity(job_description, similar_resumes, glove_model)[0]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_resume_ids = similar_resumes.iloc[sorted_indices]['ID'].tolist()
        sorted_similarity_scores = similarity_scores[sorted_indices]

        # 保存前三個結果在同一行
        result_row = {
            'Job Index': job_index + 1,
            'Company Name': company_name,
            'Position Title': position_title
        }
        for idx in range(3):
            result_row[f'Top_{idx + 1}_Resume_ID'] = sorted_resume_ids[idx] if idx < len(sorted_resume_ids) else None
            result_row[f'Top_{idx + 1}_Similarity'] = sorted_similarity_scores[idx] if idx < len(sorted_similarity_scores) else None
        
        all_results.append(result_row)
        
    # 將所有結果保存到一個CSV文件中
    results_df = pd.DataFrame(all_results)
    output_file = 'clustering_analysis/all_sorted_results.csv'
    save_sorted_results_to_csv(results_df, output_file)
    print(f"All sorted results have been saved to {output_file}.")

    # 使用 PCA 進行降維
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(features)
    
    # 繪製聚類結果
    colors = ["r", "b", "c", "y", "m"]
    x_axis = [o[0] for o in scatter_plot_points]
    y_axis = [o[1] for o in scatter_plot_points]
    plt.figure(figsize=(10, 6))
    
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