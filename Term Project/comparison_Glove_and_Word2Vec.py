import pandas as pd

# Load the CSV files
word2vec_df = pd.read_csv('clustering_analysis/word2vec_Top3.csv')
all_sorted_results_df = pd.read_csv('clustering_analysis/all_sorted_results_Glove.csv')

# Filter out 'Average' row from all_sorted_results_df
all_sorted_results_df = all_sorted_results_df[all_sorted_results_df["Job Description Index"] != "Average"]

def calculate_matching_proportion(word2vec_df, all_sorted_results_df):
    total_count = len(word2vec_df)
    total_row_match_count = 0
    
    for i in range(total_count):
        word2vec_row = word2vec_df.iloc[i]
        sorted_result_row = all_sorted_results_df.iloc[i]
        
        row_match_count = 0
        if (word2vec_row['Top_1_Resume_ID'] in ([sorted_result_row['Top_1_Resume_ID'], sorted_result_row['Top_2_Resume_ID'], sorted_result_row['Top_3_Resume_ID']])):
            row_match_count += 1
        if (word2vec_row['Top_2_Resume_ID'] in ([sorted_result_row['Top_1_Resume_ID'], sorted_result_row['Top_2_Resume_ID'], sorted_result_row['Top_3_Resume_ID']])):
            row_match_count += 1
        if (word2vec_row['Top_3_Resume_ID'] in ([sorted_result_row['Top_1_Resume_ID'], sorted_result_row['Top_2_Resume_ID'], sorted_result_row['Top_3_Resume_ID']])):
            row_match_count += 1
        print(row_match_count / 3)
        total_row_match_count += row_match_count / 3
    
    row_ratio_average = total_row_match_count / total_count
    return row_ratio_average

row_ratio_average = calculate_matching_proportion(word2vec_df, all_sorted_results_df)
print("-------------->", row_ratio_average)