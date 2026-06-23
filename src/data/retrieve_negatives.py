import pandas as pd
import numpy as np
import requests
import time
import json
import re
from itertools import zip_longest
import warnings
from collections import Counter
from pymongo import MongoClient
import csv

def normalize_passage(passage):
    """Normalize a passage by converting to lowercase, removing extra spaces, punctuation, and hidden characters."""
    passage = passage.lower()
    passage = re.sub(r'\s+', ' ', passage)
    passage = re.sub(r'[^\w\s]', '', passage)
    passage = passage.strip()
    passage = passage.replace('\u00A0', ' ')
    passage = passage.replace('\t', ' ')
    passage = passage.replace('\n', ' ')
    return passage

def main():
    start_time = time.time()

    # Connect to MongoDB
    client = MongoClient("mongodb://sibils-mongodb.lan.text-analytics.ch:27017/")
    db = client["sibils_v4_2"]
    collection = db["med25_r1_v5.5_passages"]

    # Query to find documents with an empty interaction_form
    query = {'interaction_form': {'$size': 0}}

    # Get the total number of documents in the collection
    total_documents = collection.count_documents({})

    # Calculate 10% of the total documents
    limit = int(total_documents * 0.1)

    # Set to store unique normalized passages
    unique_passages = set()

    # Iterate over the first 10% of the documents that match the query
    for doc in collection.find(query).limit(limit):
        passage = doc.get('passage', '')
        normalized_passage = normalize_passage(passage)
        unique_passages.add(normalized_passage)

    # Close the MongoDB connection
    client.close()

    # Save the unique passages to a CSV file
    output_file = 'unique_sentences.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['passage'])
        for passage in unique_passages:
            csvwriter.writerow([passage])

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total unique sentences: {len(unique_passages)}")
    print(f"Elapsed time: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
