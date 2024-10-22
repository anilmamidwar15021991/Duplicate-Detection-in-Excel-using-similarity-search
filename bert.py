import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
start_time = time.time()
threshold = 0.8  # Adjust the threshold as needed
filepath = r'C:\Users\amamidwar\OneDrive - Comscore\Documents\GITHub_Personal\Mtech2024-26\AI Project\Address_Test_Data.xlsx'
df = pd.read_excel(filepath, sheet_name='Master')

# Preprocess the addresses
def combining_all_texts(column=None):
    list_of_addresses = df[column].fillna("").tolist()  # Handle missing values
    total_words = sum(len(sentence.split()) for sentence in list_of_addresses)
    print(f"Total words: {total_words}")
    return list_of_addresses

# Load the pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # You can choose other sentence transformer models if needed

# Convert addresses to embeddings using BERT
list_of_address = combining_all_texts('Complete Address')
embeddings = model.encode(list_of_address, convert_to_tensor=False)  # Convert addresses to BERT embeddings

# Calculate cosine similarity between embeddings
similarity_matrix = cosine_similarity(embeddings)

# Output the results of duplicates based on threshold
rows_to_delete = set()

# Compare rows and mark duplicates
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if similarity_matrix[i][j] > threshold:
            print(f"Address '{df.iloc[i, 0]}' is similar to '{df.iloc[j, 0]}' with similarity score {similarity_matrix[i][j]:.2f}")
            rows_to_delete.add(j)  # Mark the duplicate row for deletion

# Remove duplicate rows
df_cleaned = df.drop(rows_to_delete)

# Save the cleaned DataFrame to a new Excel file
output_filepath = r'C:\Users\amamidwar\OneDrive - Comscore\Documents\GITHub_Personal\Mtech2024-26\AI Project\Cleaned_Address_Data_BERT.xlsx'
df_cleaned.to_excel(output_filepath, index=False)

# Log time taken
end_time = time.time()
print(f'Time taken to process: {end_time - start_time} seconds')

print(f'Duplicates removed. Cleaned data saved to: {output_filepath}')
