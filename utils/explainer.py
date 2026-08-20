import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from collections import defaultdict

TRAINING_SET = '/home/ubuntu/Desktop/progetto/training/datasets/training_set.csv'
OUTPUT_MODEL = '/home/ubuntu/Desktop/progetto/models/catboost_model.cbm'
#OUTPUT_MODEL = '/tmp/catboost_model.cbm'
FALSE_POSITIVES = '/home/ubuntu/Desktop/progetto/models/false_positives.csv'

feature_cols = ["payload_entropy_mean", "payload_entropy_cv", #"payload_size_entropy",
        "src_ip_entropy", "dst_ip_entropy", "src_mac_entropy", "dst_mac_entropy",
        "c_used", "paf", "ft_psi", "durations_entropy", "pkts_entropy", "bytes_entropy",
        "mean_duration", "cvd", "mean_pkts", "cvp", "mean_bytes", "cvb"]

model = CatBoostClassifier()
model.load_model(OUTPUT_MODEL)

# Serve un campione rappresentativo dei dati di TRAINING (non di test),
# usato da LIME per capire la distribuzione/scala di ciascuna feature
df_train = pd.read_csv(TRAINING_SET)
X_train = df_train[feature_cols].values

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_cols,
    class_names=["Normale", "Attacco"],
    mode="classification",
    discretize_continuous=True,   # LIME funziona meglio discretizzando feature continue in bin
)

# CatBoost vuole predict_proba, non predict, per dare a LIME
# le probabilita' di entrambe le classi (necessario per il suo algoritmo)
def predict_proba_wrapper(data):
    return model.predict_proba(data)

feature_impact_totals = defaultdict(float)
feature_impact_counts = defaultdict(int)

false_positives = pd.read_csv(FALSE_POSITIVES)

for idx in range(len(false_positives)):
    instance = false_positives[feature_cols].iloc[idx].values

    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_proba_wrapper,
        num_features=len(feature_cols),
        top_labels=1,
    )

    for feature_condition, weight in explanation.as_list():
        # estrai il nome della feature dalla stringa di condizione
        # (LIME restituisce es. "c_used <= 12.50", serve il nome puro)
        feature_name = next((f for f in feature_cols if f in feature_condition), feature_condition)
        feature_impact_totals[feature_name] += weight
        feature_impact_counts[feature_name] += 1

print("\nImpatto medio per feature sui falsi positivi:")
for feature in feature_cols:
    if feature_impact_counts[feature] > 0:
        avg_impact = feature_impact_totals[feature] / feature_impact_counts[feature]
        print(f"  {feature}: {avg_impact:+.4f} (su {feature_impact_counts[feature]} istanze)")