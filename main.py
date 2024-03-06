from transformers import TFPegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd

# Load the Pegasus model and tokenizer
model_name = "google/pegasus-large"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = TFPegasusForConditionalGeneration.from_pretrained(model_name)

# Load your dataset from an Excel file into a pandas DataFrame
data = pd.read_excel('data.xlsx')

# Function to generate summary using Pegasus model
def generate_summary(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=100, num_beams=5, length_penalty=0.6, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Generate summaries for each row in the dataset
generated_summaries = []
for index, row in data.iterrows():
    input_text = f"{row['Headlines']}. {row['Date']}. {row['Content']}. {row['Name of the Publication']}"
    summary = generate_summary(input_text)
    generated_summaries.append(summary)

# Add the generated summaries to the DataFrame
data['Generated Summary'] = generated_summaries

# Save the DataFrame with generated summaries to a new Excel file
data.to_excel('generated_summaries.xlsx', index=False)
