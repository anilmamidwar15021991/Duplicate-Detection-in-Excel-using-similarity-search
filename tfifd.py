import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data set
start_time = time.time()
threshold = 0.8
filepath = r'C:\Users\amamidwar\OneDrive - Comscore\Documents\GITHub_Personal\Mtech2024-26\AI Project\Address_Test_Data.xlsx'
df = pd.read_excel(filepath, sheet_name='Master')

# Convert addresses to a list of strings
def combining_all_texts(column=None):
    list_of_addresses = df[column].fillna("").tolist()  # Handle missing values
    total_words = sum(len(sentence.split()) for sentence in list_of_addresses)
    print(f"Total words: {total_words}")
    return list_of_addresses

# Create a TF-IDF vectorizer object
list_of_address = combining_all_texts('Complete Address')
vectorizer = TfidfVectorizer()
# Fit and transform the data (convert text to vectors)
vectorized_data = vectorizer.fit_transform(list_of_address)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(vectorized_data)

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
output_filepath = r'C:\Users\amamidwar\OneDrive - Comscore\Documents\GITHub_Personal\Mtech2024-26\AI Project\Cleaned_Address_Data.xlsx'
df_cleaned.to_excel(output_filepath, index=False)

# Log time taken
end_time = time.time()
print(f'Time taken to process: {end_time - start_time} seconds')

print(f'Duplicates removed. Cleaned data saved to: {output_filepath}')
