import pandas as pd

# Load the datasets
unique_no_interactions = pd.read_csv('unique_sentences.csv')
unique_3species = pd.read_csv('passages_with_3species_nointeractions.csv')
unique_random = pd.read_csv('unique_passages.csv')
#positive_passages = pd.read_csv('best_positive_sentences_3000.csv')
positive_passages = pd.read_csv('true_positives.csv')

# Keep only the first 1,000 rows for each negative dataset
unique_no_interactions = unique_no_interactions.head(1000)
unique_3species = unique_3species.head(1000)
unique_random = unique_random.head(1000)
positive_passages = positive_passages.head(3000)

# Add label columns
unique_no_interactions['label'] = 0
unique_3species['label'] = 0
unique_random['label'] = 0
positive_passages['label'] = 1

# Select only the relevant columns (assuming 'passage' or 'sentence' is the column with text)
# Replace 'passage' with 'sentence' if your column is named differently
unique_no_interactions = unique_no_interactions[['passage', 'label']]
unique_3species = unique_3species[['passage', 'label']]
unique_random = unique_random[['passage', 'label']]
positive_passages = positive_passages[['passage', 'label']]

# Concatenate the DataFrames
training_data = pd.concat([unique_no_interactions, unique_3species, unique_random, positive_passages])

# Save to CSV
training_data.to_csv('training_data_cleaned.csv', index=False)

print("Cleaned training data saved to training_data_cleaned.csv")
