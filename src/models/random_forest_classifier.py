# try out classification with a random forest classifier (emilies data)
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

start_time = time.time()

#ok so I have a list of triplets, one interaction and two species involved. I have 0 or 1 if the species are well identified, 
# 0 or 1 if the pair is interacting and in case True, 0 or 1 if the interaction is identified. Now what I want to do is build a binary classifier.

# Try reading the file with different encodings
try:
    df = pd.read_csv('biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv', sep='\t', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv('biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv', sep='\t', encoding='latin1')
    except UnicodeDecodeError:
        df = pd.read_csv('biotx-random_passages-triplets_2024-02-28_curation_EP_100original.tsv', sep='\t', encoding='cp1252')

count = df.iloc[:, 0]
id = df.iloc[:, 1]
species1_id = df.iloc[:,2]
species1_term = df.iloc[:, 3]
species1_form = df.iloc[:, 4]
species2_id = df.iloc[:, 5]
species2_term = df.iloc[:, 6]
species2_form = df.iloc[:, 7]
interaction_id = df.iloc[:, 8]
interaction_term = df.iloc[:, 9]
interaction_form = df.iloc[:, 10]
docs_count = df.iloc[:, 11]
passages_count = df.iloc[:, 12]
rank = df.iloc[:, 13]
doc_id = df.iloc[:, 14]
doc_score = df.iloc[:, 15]
passage_score = df.iloc[:, 16]
triplet_score = df.iloc[:, 17]
sentence = df.iloc[:, 18]
field = df.iloc[:, 19]
evaluation_species_identified = df.iloc[:, 20]
evaluation_pair_interacting = df.iloc[:, 21] 
comment = df.iloc[:, 23]


# Define X (features) and y (target/label)
# Use evaluation_pair_interacting as ground truth
X = df[['evaluation_species_identified']]
y = df['evaluation_pair_interacting']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the classifier
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Get detailed classification metrics
print(classification_report(y_test, y_pred))


#Since we have a binary species_identified feature, the model will likely learn whether species identification correlates
#  with interaction status
#Interaction identification could also play a role in predicting interactions  

#==================================================================================================

# Save the model
joblib.dump(clf, 'interaction_classifier.pkl')





end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

"""




"""