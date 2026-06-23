# retreive positive passages
import pandas as pd
import re
import time
from pymongo import MongoClient

start_time = time.time()

# Connect to MongoDB
client = MongoClient("mongodb://sibils-mongodb.lan.text-analytics.ch:27017/")

# Access the database and collection
db = client["sibils_v4_2"]
collection = db["med25_r1_v5.5_passages"]

# Query to retrieve passages where species1, species2, and interaction_form are NOT empty
query = {
    "species1": {"$ne": ""},
    "species2": {"$ne": ""},
    "interaction_form": {"$ne": ""}
}

# Limit results to 5000 unique passages
limit = 20000

# Function to normalize passages
def normalize_passage(passage):
    passage = passage.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    passage = re.sub(r'\s+', ' ', passage)  # Replace multiple spaces with a single space
    passage = re.sub(r'[^\w\s]', '', passage)  # Remove punctuation
    passage = passage.replace('\u00A0', ' ').replace('\t', ' ').replace('\n', ' ')  # Normalize spaces
    return passage

# Set to store unique passages
unique_passages = set()

# Iterate over matching documents until we reach 5000 unique passages
for doc in collection.find(query):
    passage = doc.get("passage", "").strip()
    normalized_passage = normalize_passage(passage)

    if normalized_passage and normalized_passage not in unique_passages:
        unique_passages.add(normalized_passage)

    # Stop when we reach 5000 unique passages
    if len(unique_passages) >= limit:
        break

# Convert to DataFrame
df = pd.DataFrame(unique_passages, columns=["passage"])

# Save to CSV
df.to_csv("positive_passages3.csv", index=False)

# Close MongoDB connection
client.close()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Saved {len(df)} unique positive passages to 'positive_passages3.csv'. Elapsed time: {elapsed_time:.2f} seconds")











"""
tf.get_logger().setLevel('ERROR')
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load the interactions file into a pandas DataFrame
interactions = pd.read_csv('interactions.tsv', sep='\t', usecols=range(92), nrows=100000, low_memory=False)

col_with_first_element = {col: interactions[col].iloc[0] for col in interactions.columns}

for col in interactions.columns:
    globals()[col] = interactions[col]


"""

