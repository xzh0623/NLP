import pandas as pd

# 加載用戶上傳的CSV文件
best_matches = pd.read_csv('model_predict/best_matches.csv')
all_sorted_results = pd.read_csv('clustering_analysis/all_sorted_results_Glove.csv')
word2vec_results = pd.read_csv('clustering_analysis/word2vec_Top3.csv')

# Filter out 'Average' row from all_sorted_results_df
all_sorted_results = all_sorted_results[all_sorted_results["Job Description Index"] != "Average"]

def calculate_matching_proportion(df1, df2):
    total_count = len(df1)
    total_row_match_count = 0
    
    for i in range(total_count):
        df1_row = df1.iloc[i]
        df2_row = df2.iloc[i]
        
        row_match_count = 0
        if (df1_row['Top_1_Resume_ID'] in ([df2_row['Top_1_Resume_ID'], df2_row['Top_2_Resume_ID'], df2_row['Top_3_Resume_ID']])):
            row_match_count += 1
        if (df1_row['Top_2_Resume_ID'] in ([df2_row['Top_1_Resume_ID'], df2_row['Top_2_Resume_ID'], df2_row['Top_3_Resume_ID']])):
            row_match_count += 1
        if (df1_row['Top_3_Resume_ID'] in ([df2_row['Top_1_Resume_ID'], df2_row['Top_2_Resume_ID'], df2_row['Top_3_Resume_ID']])):
            row_match_count += 1
        print(row_match_count / 3)
        total_row_match_count += row_match_count / 3
    
    row_ratio_average = total_row_match_count / total_count
    return row_ratio_average

row_ratio_average = calculate_matching_proportion(best_matches, all_sorted_results)
print("-------------->", row_ratio_average)

# row_ratio_average = calculate_matching_proportion(word2vec_results, all_sorted_results)
# print("-------------->", row_ratio_average)