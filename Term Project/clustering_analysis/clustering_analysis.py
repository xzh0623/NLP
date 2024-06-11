import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import Word2Vec
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

        # 找出每個簇中的關鍵字
        cluster_indices = np.where(labels == i)[0]
        cluster_features = features[cluster_indices]
        keyword_counts = np.sum(cluster_features, axis=0).A1  # 確保 keyword_counts 是一維的
        keyword_indices = np.argsort(keyword_counts)[::-1][:top_n]  # 取前 top_n 個關鍵字

        # 確保索引在特徵名稱的範圍內，並且只選擇前 top_n 個關鍵字
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

def find_most_similar_cluster_word2vec(job_description, cluster_centers, model):
    job_desc_features = np.mean([model.wv[word] for word in job_description.split() if word in model.wv] or [np.zeros(100)], axis=0).reshape(1, -1)
    cluster_similarities = cosine_similarity(job_desc_features, cluster_centers)
    most_similar_cluster_index = np.argmax(cluster_similarities)
    return most_similar_cluster_index, cluster_similarities

def find_most_similar_resumes(job_description, similar_cluster_indices, resume_data, vectorizer, method_name):
    similar_resumes = resume_data.iloc[similar_cluster_indices].dropna(subset=['Processed_Resume_str'])
    similarity_scores = calculate_similarity(job_description, similar_resumes['Processed_Resume_str'], vectorizer)[0]
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_resume_ids = similar_resumes.iloc[sorted_indices]['ID'].tolist()
    sorted_similarity_scores = similarity_scores[sorted_indices]
    most_similar_resume_id = sorted_resume_ids[0]
    
    return most_similar_resume_id, sorted_resume_ids, sorted_similarity_scores

def find_most_similar_resumes_word2vec(job_description, similar_cluster_indices, resume_data, model):
    similar_resumes = resume_data.iloc[similar_cluster_indices].dropna(subset=['Processed_Resume_str'])
    similarity_scores = [cosine_similarity(
        np.mean([model.wv[word] for word in job_description.split() if word in model.wv] or [np.zeros(100)], axis=0).reshape(1, -1),
        np.mean([model.wv[word] for word in resume.split() if word in model.wv] or [np.zeros(100)], axis=0).reshape(1, -1)
    )[0][0] for resume in similar_resumes['Processed_Resume_str']]
    
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_resume_ids = similar_resumes.iloc[sorted_indices]['ID'].tolist()
    sorted_similarity_scores = [similarity_scores[i] for i in sorted_indices]
    most_similar_resume_id = sorted_resume_ids[0]
    
    return most_similar_resume_id, sorted_resume_ids, sorted_similarity_scores

def main():
    # 加載預處理生成的特徵文件
    tfidf_features_path = 'data_processing/test_data/TfidfVectorizerFeature_Resume_str.npz'
    count_features_path = 'data_processing/test_data/CountVectorizerFeature_Resume_str.npz'
    word2vec_path = 'data_processing/test_data/word2vecModel_Resume_str.w2v'
    vectorizer_path = 'data_processing/test_data/TfidfVectorizerVectorizer_Resume_str.pkl'
    count_vectorizer_path = 'data_processing/test_data/CountVectorizerVectorizer_Resume_str.pkl'
    job_vectorizer_path = 'data_processing/training_data/TfidfVectorizerVectorizer_model_response.pkl'

    # 加載特徵
    tfidf_features = load_features(tfidf_features_path)
    count_features = load_features(count_features_path)

    # 加載向量化器和模型
    vectorizer = joblib.load(vectorizer_path)
    count_vectorizer = joblib.load(count_vectorizer_path)
    word2vec_model = Word2Vec.load(word2vec_path)
    job_vectorizer = joblib.load(job_vectorizer_path)
    feature_names = vectorizer.get_feature_names_out()
    count_feature_names = count_vectorizer.get_feature_names_out()

    # 打印特徵名稱數組的長度
    print(f"TF-IDF feature names length: {len(feature_names)}")
    print(f"Count feature names length: {len(count_feature_names)}")

    # 選擇要使用的特徵進行聚類
    num_clusters = 5

    def perform_clustering_and_print(features, feature_names, method_name):
        # 使用 K-means 進行聚類
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_

        # 解释簇中心和关键字
        cluster_centers, cluster_keywords = interpret_clusters(features, labels, num_clusters, feature_names)
        cluster_centers_df = pd.DataFrame(cluster_centers)
        cluster_centers_df.to_csv(f'clustering_analysis/{method_name}_cluster_centers.csv', index=False)
        print(f"{method_name} cluster centers have been saved to '{method_name}_cluster_centers.csv'.")

        # 打印每個簇的關鍵字
        for i, keywords in enumerate(cluster_keywords):
            print(f"{method_name} Cluster {i} keywords: {', '.join(keywords)}")

        return labels, cluster_centers

    ### TF-IDF 聚类
    tfidf_labels, tfidf_cluster_centers = perform_clustering_and_print(tfidf_features, feature_names, "TF-IDF")
    print("TF-IDF Clustering Results:")
    for i in range(num_clusters):
        cluster_indices = np.where(tfidf_labels == i)[0]
        print(f"Cluster {i} contains {len(cluster_indices)} resumes")

    # CountVectorizer 聚类
    count_labels, count_cluster_centers = perform_clustering_and_print(count_features, count_feature_names, "CountVectorizer")
    print("CountVectorizer Clustering Results:")
    for i in range(num_clusters):
        cluster_indices = np.where(count_labels == i)[0]
        print(f"Cluster {i} contains {len(cluster_indices)} resumes")

    # Word2Vec 聚类
    resume_data = pd.read_csv('dataset/processed_testData.csv')
    resume_sentences = resume_data['Processed_Resume_str'].fillna('').apply(lambda x: x.split())
    resume_vectors = [np.mean([word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for sentence in resume_sentences]
    resume_vectors = np.array(resume_vectors)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(resume_vectors)
    word2vec_labels = kmeans.labels_
    word2vec_cluster_centers = kmeans.cluster_centers_
    # 無法直接獲取關鍵詞，因為 Word2Vec 的聚類結果是基於向量
    print("Word2Vec Clustering Results:")
    for i in range(num_clusters):
        cluster_indices = np.where(word2vec_labels == i)[0]
        print(f"Cluster {i} contains {len(cluster_indices)} resumes")

    ### 加載職位描述數據
    job_descriptions_df = pd.read_csv('dataset/processed_trainingData.csv')
    results = []
    word2vec_results = []

    # 初始化變量來計算平均相似度
    total_similarity_tfidf = 0
    total_similarity_count = 0
    total_similarity_word2vec = 0      
        
    for job_index in range(len(job_descriptions_df)):
    # for job_index in range(10):
        job_description = job_descriptions_df['Processed_model_response'].iloc[job_index]
        company_name = job_descriptions_df.iloc[job_index]['company_name']
        position_title = job_descriptions_df.iloc[job_index]['position_title']

        # 找到與職位描述最相似的 TF-IDF 聚类簇
        most_similar_cluster_index_tfidf, cluster_similarities_tfidf = find_most_similar_cluster(job_description, tfidf_cluster_centers, vectorizer)
        similar_cluster_indices_tfidf = np.where(tfidf_labels == most_similar_cluster_index_tfidf)[0]
        tfidf_similarity_score = cluster_similarities_tfidf[0][most_similar_cluster_index_tfidf]
        most_similar_resume_tfidf, sorted_resume_ids_tfidf, sorted_similarity_scores_tfidf = find_most_similar_resumes(job_description, similar_cluster_indices_tfidf, resume_data, vectorizer, "TF-IDF")

        # 找到與職位描述最相似的 CountVectorizer 聚类簇
        most_similar_cluster_index_count, cluster_similarities_count = find_most_similar_cluster(job_description, count_cluster_centers, count_vectorizer)
        similar_cluster_indices_count = np.where(count_labels == most_similar_cluster_index_count)[0]
        count_similarity_score = cluster_similarities_count[0][most_similar_cluster_index_count]
        most_similar_resume_count, sorted_resume_ids_count, sorted_similarity_scores_count = find_most_similar_resumes(job_description, similar_cluster_indices_count, resume_data, count_vectorizer, "CountVectorizer")

        # 找到與職位描述最相似的 Word2Vec 聚类簇
        most_similar_cluster_index_word2vec, cluster_similarities_word2vec = find_most_similar_cluster_word2vec(job_description, word2vec_cluster_centers, word2vec_model)
        similar_cluster_indices_word2vec = np.where(word2vec_labels == most_similar_cluster_index_word2vec)[0]
        word2vec_similarity_score = cluster_similarities_word2vec[0][most_similar_cluster_index_word2vec]
        most_similar_resume_word2vec, sorted_resume_ids_word2vec, sorted_similarity_scores_word2vec = find_most_similar_resumes_word2vec(job_description, similar_cluster_indices_word2vec, resume_data, word2vec_model)

        # 累加相似度
        total_similarity_tfidf += tfidf_similarity_score
        total_similarity_count += count_similarity_score
        total_similarity_word2vec += word2vec_similarity_score

        print(job_index+1, "/", len(job_descriptions_df))
        results.append({
            "Job Description Index": job_index+1,
            'Company Name': company_name,
            'Position Title': position_title,
            'TF-IDF Most similar cluster index': most_similar_cluster_index_tfidf,
            'CountVectorizer Most similar cluster index': most_similar_cluster_index_count,
            'Word2Vec Most similar cluster index': most_similar_cluster_index_word2vec,
            "TF-IDF Most Similarity Cluster Scores": tfidf_similarity_score,
            "CountVectorizer Most Similarity Cluster Scores": count_similarity_score,
            "Word2Vec Most Similarity Cluster Scores": word2vec_similarity_score,
            "Most Similar Resume ID (TF-IDF)": most_similar_resume_tfidf,
            "Most Similar Resume Score (TF-IDF)": sorted_similarity_scores_tfidf[0],
            "Most Similar Resume ID (CountVectorizer)": most_similar_resume_count,
            "Most Similar Resume Score (CountVectorizer)": sorted_similarity_scores_count[0]
        })

        word2vec_results.append({
            "Job Description Index": job_index+1,
            'Company Name': company_name,
            'Position Title': position_title,
            "Top_1_Resume_ID": most_similar_resume_word2vec,
            "Top_1_Similarity": sorted_similarity_scores_word2vec[0],
            "Top_2_Resume_ID": sorted_resume_ids_word2vec[1],
            "Top_2_Similarity": sorted_similarity_scores_word2vec[1],
            "Top_3_Resume_ID": sorted_resume_ids_word2vec[2],
            "Top_3_Similarity": sorted_similarity_scores_word2vec[2]
        })

    # 計算平均相似度
    average_similarity_tfidf = total_similarity_tfidf / len(job_descriptions_df)
    average_similarity_count = total_similarity_count / len(job_descriptions_df)
    average_similarity_word2vec = total_similarity_word2vec / len(job_descriptions_df)

    # 添加平均相似度到結果中
    results.insert(0, {
        "Job Description Index": "Average",
        'Company Name': "",
        'Position Title': "",
        'TF-IDF Most similar cluster index': "",
        'CountVectorizer Most similar cluster index': "",
        'Word2Vec Most similar cluster index': "",
        "TF-IDF Most Similarity Cluster Scores": average_similarity_tfidf,
        "CountVectorizer Most Similarity Cluster Scores": average_similarity_count,
        "Word2Vec Most Similarity Cluster Scores": average_similarity_word2vec,
        "Most Similar Resume ID (TF-IDF)": "",
        "Most Similar Resume Score (TF-IDF)": "",
        "Most Similar Resume ID (CountVectorizer)": "",
        "Most Similar Resume Score (CountVectorizer)": "",
        "Most Similar Resume ID (Word2Vec)": "",
        "Most Similar Resume Score (Word2Vec)": ""
    })

    results_df = pd.DataFrame(results)
    results_df.to_csv('clustering_analysis/similarity_scores(max in clusters).csv', index=False)
    print("Similarity scores have been saved to 'clustering_analysis/similarity_scores(max in clusters).csv'.")

    results_df = pd.DataFrame(word2vec_results)
    results_df.to_csv('clustering_analysis/word2vec_Top3.csv', index=False)
    print("Similarity scores have been saved to 'clustering_analysis/word2vec_Top3.csv'.")


    ### 使用 PCA 進行降維並繪制聚類結果
    def plot_clusters(features, labels, method_name):
        pca = PCA(n_components=2)
        scatter_plot_points = pca.fit_transform(features.toarray())

        colors = ["r", "b", "c", "y", "m"]
        x_axis = [o[0] for o in scatter_plot_points]
        y_axis = [o[1] for o in scatter_plot_points]
        plt.figure(figsize=(10, 6))

        for i in range(num_clusters):
            cluster_points = np.array([scatter_plot_points[j] for j in range(len(labels)) if labels[j] == i])
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}')

        plt.title(f"{method_name} Clustered Resumes and Job Descriptions")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.show()

    plot_clusters(tfidf_features, tfidf_labels, "TF-IDF")
    plot_clusters(count_features, count_labels, "CountVectorizer")

    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(resume_vectors)
    x_axis = scatter_plot_points[:, 0]
    y_axis = scatter_plot_points[:, 1]
    plt.figure(figsize=(10, 6))

    # 定義顏色
    colors = ["r", "b", "c", "y", "m"]

    for i in range(num_clusters):
        cluster_points = scatter_plot_points[word2vec_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}')

    plt.title("Word2Vec Clustered Resumes and Job Descriptions")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

