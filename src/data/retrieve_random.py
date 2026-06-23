# Description: This script retrieves a specified number of random passages
#  from the PMC25 dataset.

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
import random

start_time = time.time()

client = MongoClient("mongodb://sibils-mongodb.lan.text-analytics.ch:27017/") 
db = client["sibils_v4_2"]
collection = db["pmc25_r1_v5.5"]

num_sentences_needed = 3000

try:

    unique_sentences = set()
    retrieved_count = 0
    articles_processed = 0

    articles_cursor = collection.find().sort("_id")

    for article in articles_cursor:
        articles_processed += 1

        try:
            if "sentences" in article and isinstance(article["sentences"], list):
                for sentence_data in article["sentences"]:
                    if (
                        isinstance(sentence_data, dict)
                        and "field" in sentence_data
                        and sentence_data["field"] == "text"
                        and "sentence" in sentence_data
                    ):
                        sentence = sentence_data["sentence"]
                        if sentence not in unique_sentences:
                            unique_sentences.add(sentence)
                            retrieved_count += 1
                            if retrieved_count % 100 == 0:
                                print(f"Retrieved {retrieved_count} unique sentences...")
                            if retrieved_count >= num_sentences_needed:
                                break
            if retrieved_count >= num_sentences_needed:
                break

        except Exception as e:
            print(f"An error occurred while processing an article: {e}")
            continue

    print(f"Processed {articles_processed} articles.")

except pymongo.errors.ConnectionFailure as e:
    print(f"MongoDB Connection Failure: {e}")
except Exception as e:
    print(f"An unexpected error occurred during connection or setup: {e}")

finally:
    if client:
        client.close()

sentences = list(unique_sentences)

if sentences:
    print(f"Successfully retrieved {len(sentences)} unique sentences.")

    with open("unique_passages.csv", "w", encoding="utf-8", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["passage"])  # Write header
        for sentence in sentences:
            csvwriter.writerow([sentence])

else:
    print("Failed to retrieve sentences.")


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")