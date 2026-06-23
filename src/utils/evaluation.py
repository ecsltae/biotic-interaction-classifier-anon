#evaluate_nlp1.py liked to nlp_solution1.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the results
results_df = pd.read_csv('rule_based_nlp_results.csv')

# Extract true and predicted labels
true_labels = results_df['true_label']
predicted_labels = results_df['predicted_label']

# Generate classification report
classification_rep = classification_report(true_labels, predicted_labels, target_names=['Negative', 'Positive'])
print("Classification Report:\n", classification_rep)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Accuracy: {accuracy:.4f}")

print(f"Evaluation complete. Confusion matrix saved to 'confusion_matrix.png'.")
