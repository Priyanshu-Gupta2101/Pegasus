import pandas as pd
from rouge import Rouge

# Load your dataset from an Excel file into a pandas DataFrame
data = pd.read_excel('data.xlsx')

# Load the DataFrame with generated summaries from an Excel file
generated_data = pd.read_excel('generated_summaries.xlsx')

# Function to calculate ROUGE scores
def calculate_rouge_scores(generated_summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summary, reference_summary)
    return scores

# Calculate ROUGE scores for each pair of generated and reference summaries
rouge_scores = []
for index, row in generated_data.iterrows():
    rouge_score = calculate_rouge_scores(row['Generated Summary'], row['Summary'])
    rouge_scores.append(rouge_score)

# Extract ROUGE-1 (unigram) recall scores
rouge_1_recall_scores = [score[0]['rouge-1']['r'] for score in rouge_scores]

# Calculate average ROUGE-1 recall score
average_rouge_1_recall = sum(rouge_1_recall_scores) / len(rouge_1_recall_scores)

print(f"Average ROUGE-1 Recall: {average_rouge_1_recall}")
