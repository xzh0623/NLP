import pandas as pd

# 加載用戶上傳的CSV文件
best_matches_path = 'model_predict/best_matches.csv'
all_sorted_results_path = 'clustering_analysis/all_sorted_results.csv'

best_matches = pd.read_csv(best_matches_path)
all_sorted_results = pd.read_csv(all_sorted_results_path)

# # 顯示加載的CSV文件的頭部，以檢查數據格式
# print("Best Matches CSV:")
# print(best_matches.head())
# print("\nAll Sorted Results CSV:")
# print(all_sorted_results.head())

# 計算每列的Top_1_Resume_ID, Top_2_Resume_ID, Top_3_Resume_ID是否相同，並給出相同的比例
def calculate_matching_proportion(best_matches, all_sorted_results):
    match_count = 0
    total_count = len(best_matches)
    total_row_match_count = 0
    
    for i in range(total_count):
        best_match_row = best_matches.iloc[i]
        sorted_result_row = all_sorted_results.iloc[i]
        
        if (best_match_row['Top_1_Resume_ID'] == sorted_result_row['Top_1_Resume_ID'] and
            best_match_row['Top_2_Resume_ID'] == sorted_result_row['Top_2_Resume_ID'] and
            best_match_row['Top_3_Resume_ID'] == sorted_result_row['Top_3_Resume_ID']):
            match_count += 1
        
        row_match_count = 0
        if (best_match_row['Top_1_Resume_ID'] == sorted_result_row['Top_1_Resume_ID']):
            row_match_count += 1
        if (best_match_row['Top_2_Resume_ID'] == sorted_result_row['Top_2_Resume_ID']):
            row_match_count += 1
        if (best_match_row['Top_3_Resume_ID'] == sorted_result_row['Top_3_Resume_ID']):
            row_match_count += 1
        print(row_match_count / 3)
        total_row_match_count += row_match_count / 3
    
    proportion = match_count / total_count
    row_ratio_average = total_row_match_count / total_count
    return proportion, row_ratio_average

proportion, row_ratio_average = calculate_matching_proportion(best_matches, all_sorted_results)
print("-------------->", proportion, row_ratio_average)
