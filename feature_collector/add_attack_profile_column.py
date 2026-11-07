"""
Aggiunge (o aggiorna) la colonna con il profilo di attacco ai dataset usati
per il training, leggendo il profilo di ciascuno scenario da
network_settings.csv.

network_settings.csv ha una riga per scenario, nell'ordine:
    topo,hard_timeout,idle_timeout,flow_table_size,npkts,profile,anp,att_duration,fill_time
e lo scenario_id e' il numero di riga (posizione), non una colonna esplicita.

Per ogni dataset (dataset_ftf.csv, dataset_mfr.csv, ...):
- le righe con label == 0 (traffico benigno) ricevono BENIGN_PROFILE_LABEL,
- le righe con label == 1 (traffico malevolo) ricevono il valore di
  network_settings.csv["profile"] dello scenario corrispondente.

Prima di sovrascrivere ogni dataset viene salvato un backup <file>.bak
(solo se non esiste gia').
"""

import os
import shutil
import numpy as np
import pandas as pd

# Modules loader
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

NETWORK_SETTINGS_PATH = current+'/datasets/network_settings.csv'

# Un dataset per riga: (percorso, colonna scenario, colonna label)
DATASETS = [
    (current+'/datasets/dataset_ftf.csv', 'scenario_id', 'label'),
    #(current+'/datasets/dataset_mfr.csv', 'scenario_id', 'label'),
]

PROFILE_COL_IN_SETTINGS = 'profile'
ATTACK_PROFILE_COL = 'attack_profile'   # deve combaciare con ATTACK_PROFILE_COL negli script di training
BENIGN_PROFILE_LABEL = 'benigni'        # deve combaciare con BENIGN_PROFILE_LABEL negli script di training

# Scenario_id del primo scenario in network_settings.csv: 0 se gli
# scenario_id nei dataset partono da 0 (indice di riga standard, escluso
# l'header), 1 se partono da 1. Verifica con l'output diagnostico qui sotto
# e correggi se necessario.
SCENARIO_ID_BASE = 0

DRY_RUN = False  # se True, calcola e stampa tutto ma non scrive i file

def build_profile_map(settings_path, profile_col, scenario_id_base):
    df_settings = pd.read_csv(settings_path)
    scenario_ids = df_settings.index + scenario_id_base
    profile_map = dict(zip(scenario_ids, df_settings[profile_col]))
    print(f"Caricati {len(profile_map)} scenari da {settings_path} "
          f"(scenario_id da {min(profile_map)} a {max(profile_map)})")
    print(df_settings[profile_col].value_counts())
    return profile_map

def add_attack_profile_column(dataset_path, scenario_col, label_col, profile_map):
    print(f"\n--- {dataset_path} ---")
    df = pd.read_csv(dataset_path)

    dataset_scenarios = set(df[scenario_col].unique())
    missing = dataset_scenarios - set(profile_map.keys())
    if missing:
        example = sorted(missing)[:10]
        raise ValueError(
            f"{len(missing)} scenari presenti in {dataset_path} non hanno un "
            f"corrispondente in {NETWORK_SETTINGS_PATH} (es. {example}). "
            f"Controlla SCENARIO_ID_BASE (attualmente {SCENARIO_ID_BASE})."
        )

    profiles = df[scenario_col].map(profile_map)
    df[ATTACK_PROFILE_COL] = np.where(df[label_col] == 0, BENIGN_PROFILE_LABEL, profiles)

    # Controllo di coerenza: ogni scenario di attacco deve avere un solo
    # profilo non-benigno associato alle sue righe malevole.
    attack_rows = df[df[label_col] != 0]
    profiles_per_scenario = attack_rows.groupby(scenario_col)[ATTACK_PROFILE_COL].nunique()
    inconsistent = profiles_per_scenario[profiles_per_scenario > 1]
    if len(inconsistent) > 0:
        print(f"[!] Attenzione: {len(inconsistent)} scenari hanno piu' di un "
              f"profilo di attacco tra le righe malevole: {list(inconsistent.index)[:10]}")

    print(f"Distribuzione '{ATTACK_PROFILE_COL}' in {os.path.basename(dataset_path)}:")
    print(df[ATTACK_PROFILE_COL].value_counts())

    if DRY_RUN:
        print("[DRY_RUN] Nessun file scritto.")
        return

    backup_path = dataset_path + '.bak'
    if not os.path.exists(backup_path):
        shutil.copy2(dataset_path, backup_path)
        print(f"Backup salvato in {backup_path}")
    else:
        print(f"Backup gia' presente in {backup_path}, non sovrascritto")

    df.to_csv(dataset_path, index=False)
    print(f"Colonna '{ATTACK_PROFILE_COL}' scritta in {dataset_path}")


if __name__ == "__main__":
    profile_map = build_profile_map(
        NETWORK_SETTINGS_PATH, PROFILE_COL_IN_SETTINGS, SCENARIO_ID_BASE
    )

    for dataset_path, scenario_col, label_col in DATASETS:
        add_attack_profile_column(dataset_path, scenario_col, label_col, profile_map)
