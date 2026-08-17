import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

TRAINING_SET = '/home/ubuntu/Desktop/progetto/training/datasets/training_set.csv'
OUTPUT_MODEL = '/tmp/catboost_model.cbm'

# Load the dataset
df = pd.read_csv(TRAINING_SET)

# Prepare the features and target
X = df[['payload_entropy_mean','payload_entropy_cv','payload_size_entropy',
'src_ip_entropy','dst_ip_entropy','src_mac_entropy',
'dst_mac_entropy','c_used','paf','ft_psi','durations_entropy',
'pkts_entropy','bytes_entropy','mean_duration','cvd','mean_pkts','cvp','mean_bytes','cvb']]
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the CatBoost classifier
catboost_classifier = CatBoostClassifier(
    silent=True,
    random_state=42,
    auto_class_weights='Balanced'
)

# Train the classifier
catboost_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = catboost_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the model
catboost_classifier.save_model(OUTPUT_MODEL)
