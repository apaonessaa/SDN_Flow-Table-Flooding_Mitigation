import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, classification_report

TRAINING_SET = '/home/ubuntu/Desktop/progetto/training/datasets/training_set.csv'
OUTPUT_MODEL = '/home/ubuntu/Desktop/progetto/models/catboost_model.cbm'
#OUTPUT_MODEL = '/tmp/catboost_model.cbm'

# Load the dataset
df = pd.read_csv(TRAINING_SET)

# Prepare the features and target
X = df[['payload_entropy_mean','payload_entropy_cv',#'payload_size_entropy',
'src_ip_entropy','dst_ip_entropy','src_mac_entropy','dst_mac_entropy',
'c_used','paf','ft_psi','durations_entropy',
'pkts_entropy','bytes_entropy','mean_duration','cvd',
'mean_pkts','cvp','mean_bytes','cvb']]
y = df['label']

# Verifica che nessuno scenario domini eccessivamente il dataset
counts = df.groupby(['scenario_id', 'label']).size().unstack(fill_value=0)
print(counts)
print(f"\nDeviazione standard tra scenari: {counts.sum(axis=1).std()}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
train_pool = Pool(data=X_train, label=y_train)
test_pool = Pool(data=X_test, label=y_test)

# Initialize the CatBoost classifier
params = {
    'silent': True,
    'depth': 6,                     # Depth of the trees
    'learning_rate': 0.1,           # Learning rate of the model
    'l2_leaf_reg': 3.0,               # L2 regularization term on weights
    'rsm': 0.8,                     # Random Selection Rate
    'iterations': 100,              # Number of boosting iterations
    'loss_function': 'Logloss',  # Loss function
    'eval_metric': 'F1',      # Evaluation metric
    'random_seed': 42,              # Random seed for reproducibility
    'auto_class_weights': 'Balanced',
    #'bagging_temperature': 1.0,
    #'random_strength': 2.0,
}
catboost_classifier = CatBoostClassifier(**params)

# Train the classifier
catboost_classifier.fit(
    train_pool, 
    eval_set=test_pool, 
    verbose=50, 
    early_stopping_rounds=20,
    use_best_model=True)

# Make predictions on the test set
y_pred = catboost_classifier.predict(test_pool)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(catboost_classifier.get_feature_importance(prettified=True))

# Save the model
catboost_classifier.save_model(OUTPUT_MODEL)
print(f"Model path: {OUTPUT_MODEL}")
