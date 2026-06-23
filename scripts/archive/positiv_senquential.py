import pandas as pd

# Load the positive sentences
positive_sentences_df = pd.read_csv('positive_passages3.csv')

# Define the keywords to exclude
exclude_keywords = ['ate', 'eat', 'eated', 'eating']

# Function to check if a sentence contains any of the exclude keywords
def contains_exclude_keywords(sentence):
    sentence_lower = sentence.lower()
    return any(keyword in sentence_lower for keyword in exclude_keywords)

# Filter out sentences containing the exclude keywords
filtered_sentences_df = positive_sentences_df[~positive_sentences_df['passage'].apply(contains_exclude_keywords)]
removed_sentences_df = positive_sentences_df[positive_sentences_df['passage'].apply(contains_exclude_keywords)]

# Save the filtered sentences to a new CSV file
filtered_sentences_df.to_csv('positive_no_eat.csv', index=False)

# Save the removed sentences to a separate CSV file
removed_sentences_df.to_csv('positive_eat.csv', index=False)

print(f"Filtered sentences saved to 'positive_no_eat.csv'. Total sentences after filtering: {len(filtered_sentences_df)}")
print(f"Removed sentences saved to 'positive_eat.csv'. Total sentences removed: {len(removed_sentences_df)}")
