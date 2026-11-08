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

Uso:
    # comportamento di default (dataset_ftf.csv e dataset_mfr.csv in ./datasets)
    python3 add_attack_profile.py

    # dataset espliciti
    python3 add_attack_profile.py /path/dataset_ftf_2_10.csv /path/dataset_ftf_3_100.csv

    # network_settings.csv in un'altra posizione
    python3 add_attack_profile.py --network-settings /path/network_settings.csv /path/dataset_ftf.csv

    # prova senza scrivere nulla
    python3 add_attack_profile.py --dry-run /path/dataset_ftf.csv

    # scrive le versioni modificate in un'altra cartella, senza backup
    # (utile se i backup originali sono gia' conservati altrove)
    python3 add_attack_profile.py --output-dir /path/out --no-backup datasets/dataset_ftf_*_*.csv
"""

import os
import sys
import shutil
import argparse
import numpy as np
import pandas as pd

# Modules loader
current = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(current)
sys.path.append(root)

DEFAULT_NETWORK_SETTINGS_PATH = current + '/datasets/network_settings.csv'

# Dataset usati quando nessun path viene passato da linea di comando
DEFAULT_DATASETS = [
    current + '/datasets/dataset_ftf.csv',
    current + '/datasets/dataset_mfr.csv',
]

PROFILE_COL_IN_SETTINGS = 'profile'
ATTACK_PROFILE_COL = 'attack_profile'   # deve combaciare con ATTACK_PROFILE_COL negli script di training
BENIGN_PROFILE_LABEL = 'benigni'        # deve combaciare con BENIGN_PROFILE_LABEL negli script di training


def build_profile_map(settings_path, profile_col, scenario_id_base):
    df_settings = pd.read_csv(settings_path)
    scenario_ids = df_settings.index + scenario_id_base
    profile_map = dict(zip(scenario_ids, df_settings[profile_col]))
    print(f"Caricati {len(profile_map)} scenari da {settings_path} "
          f"(scenario_id da {min(profile_map)} a {max(profile_map)})")
    print(df_settings[profile_col].value_counts())
    return profile_map


def add_attack_profile_column(dataset_path, scenario_col, label_col, profile_map,
                               network_settings_path, scenario_id_base,
                               output_dir=None, backup=True, dry_run=False):
    print(f"\n--- {dataset_path} ---")
    df = pd.read_csv(dataset_path)

    dataset_scenarios = set(df[scenario_col].unique())
    missing = dataset_scenarios - set(profile_map.keys())
    if missing:
        example = sorted(missing)[:10]
        raise ValueError(
            f"{len(missing)} scenari presenti in {dataset_path} non hanno un "
            f"corrispondente in {network_settings_path} (es. {example}). "
            f"Controlla --scenario-id-base (attualmente {scenario_id_base})."
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

    # Path di destinazione: stesso file (overwrite) oppure stesso nome dentro
    # output_dir, se specificato.
    if output_dir:
        out_path = os.path.join(output_dir, os.path.basename(dataset_path))
    else:
        out_path = dataset_path

    if dry_run:
        print(f"[DRY_RUN] Nessun file scritto (destinazione sarebbe: {out_path}).")
        return

    # Il backup ha senso solo quando si sovrascrive il file originale: se si
    # scrive altrove (output_dir diverso), l'originale resta comunque intatto.
    if backup and out_path == dataset_path:
        backup_path = dataset_path + '.bak'
        if not os.path.exists(backup_path):
            shutil.copy2(dataset_path, backup_path)
            print(f"Backup salvato in {backup_path}")
        else:
            print(f"Backup gia' presente in {backup_path}, non sovrascritto")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Colonna '{ATTACK_PROFILE_COL}' scritta in {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggiunge la colonna attack_profile ai dataset FTF/MFR.",
    )
    parser.add_argument(
        'datasets', nargs='*', default=DEFAULT_DATASETS,
        help=f"Path dei dataset CSV da elaborare (default: {DEFAULT_DATASETS})",
    )
    parser.add_argument(
        '--network-settings', default=DEFAULT_NETWORK_SETTINGS_PATH,
        help=f"Path di network_settings.csv (default: {DEFAULT_NETWORK_SETTINGS_PATH})",
    )
    parser.add_argument(
        '--scenario-col', default='scenario_id',
        help="Nome della colonna scenario nei dataset (default: scenario_id)",
    )
    parser.add_argument(
        '--label-col', default='label',
        help="Nome della colonna label nei dataset (default: label)",
    )
    parser.add_argument(
        '--scenario-id-base', type=int, default=0,
        help="scenario_id del primo scenario in network_settings.csv: 0 se gli "
             "scenario_id nei dataset partono da 0, 1 se partono da 1 (default: 0)",
    )
    parser.add_argument(
        '--output-dir', default=None,
        help="Cartella in cui scrivere i dataset modificati (stesso nome file "
             "dell'originale). Se omesso, sovrascrive il file di partenza.",
    )
    parser.add_argument(
        '--no-backup', action='store_true',
        help="Non creare il file <dataset>.bak prima di sovrascrivere. "
             "Ignorato se --output-dir e' specificato (l'originale non viene "
             "comunque toccato).",
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help="Calcola e stampa tutto ma non scrive i file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for path in args.datasets:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset non trovato: {path}")
    if not os.path.exists(args.network_settings):
        raise FileNotFoundError(f"network_settings.csv non trovato: {args.network_settings}")

    profile_map = build_profile_map(
        args.network_settings, PROFILE_COL_IN_SETTINGS, args.scenario_id_base
    )

    for dataset_path in args.datasets:
        add_attack_profile_column(
            dataset_path, args.scenario_col, args.label_col, profile_map,
            args.network_settings, args.scenario_id_base,
            output_dir=args.output_dir, backup=not args.no_backup, dry_run=args.dry_run,
        )