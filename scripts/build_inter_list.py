import json
import pandas as pd

# Load the JSON file
with open('robiext_v2025.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize a set to store unique interaction terms
interaction_terms = set()

# Iterate through each concept in the JSON data
for concept in data['concepts']:
    # Check if the preferred term is relevant
    if concept['preferred_term']['relevance']:
        interaction_terms.add(concept['preferred_term']['term'])

    # Check each synonym for relevance
    for synonym in concept['synonyms']:
        if synonym['relevance']:
            interaction_terms.add(synonym['term'])

# Convert the set to a DataFrame
interaction_df = pd.DataFrame({'interaction': list(interaction_terms)})

# Save the DataFrame to a CSV file
interaction_df.to_csv('interaction_dict.csv', index=False)

print("interaction_dict.csv has been created successfully!")
