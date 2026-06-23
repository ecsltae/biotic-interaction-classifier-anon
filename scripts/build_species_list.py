import json
import pandas as pd

# Load the JSON file
with open('ott_v3.7.2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize a set to store unique species terms
species_terms = set()

# Iterate through each concept in the JSON data
for concept in data['concepts']:
    # Check if the preferred term is relevant
    if concept['preferred_term']['relevance']:
        species_terms.add(concept['preferred_term']['term'])

    # Check each synonym for relevance
    for synonym in concept['synonyms']:
        if synonym['relevance']:
            species_terms.add(synonym['term'])

# Convert the set to a DataFrame
species_df = pd.DataFrame({'species': list(species_terms)})

# Save the DataFrame to a CSV file
species_df.to_csv('species_dict.csv', index=False)

print("species_dict.csv has been created successfully!")
