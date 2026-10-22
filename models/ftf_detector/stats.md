# Data preproccesing: Quantile

```txt
Numero di entry del dataset: 9615
label          0    1
scenario_id          
0            237  249
1            178  176
2            118  111
3            161  162
4            824  759
5            193  180
6            776  776
7            254  247
8             96   93
9            170  163
10           226  223
11           157  155
12           180  177
13           221  218
14           169  167
15           900  899

Deviazione standard tra scenari: 526.6252897775926
[!] Stratificazione per 'attack_profile' non applicabile (The test_size = 2 should be greater or equal to the number of classes = 3); eseguo uno split non stratificato.


=== Tuning XGBoost ===
Best trial: 7. Best value: 0.833328: 100%|███████████████████████████████████████████████████████| 10/10 [00:26<00:00,  2.67s/it]

[XGBoost] Miglior F1 medio in CV: 0.8333
[XGBoost] Migliori parametri: {'max_depth': 4, 'learning_rate': 0.15242391728466367, 'reg_lambda': 4.622919742400237, 'colsample_bytree': 0.831261142176991, 'subsample': 0.6558555380447055, 'min_child_weight': 5.680612190600297}


=== Tuning CatBoost ===
Best trial: 9. Best value: 0.841278: 100%|███████████████████████████████████████████████████████| 10/10 [00:31<00:00,  3.18s/it]

[CatBoost] Miglior F1 medio in CV: 0.8413
[CatBoost] Migliori parametri: {'depth': 8, 'learning_rate': 0.15826541904647565, 'l2_leaf_reg': 2.2388950287268727, 'rsm': 0.5979914312095727, 'bagging_temperature': 0.1356818667316142, 'random_strength': 1.6266516538163218}


=== Tuning FeatBERT ===
Best trial: 7. Best value: 0.842757: 100%|██████████████████████████████████████████████████████| 10/10 [58:29<00:00, 350.98s/it]

[FeatBERT] Miglior F1 medio in CV: 0.8428
[FeatBERT] Migliori parametri: {'num_attention_heads': 8, 'head_dim': 24, 'num_hidden_layers': 2, 'intermediate_size': 128, 'dropout': 0.43573029509385885, 'learning_rate': 0.00040489662225846743, 'weight_decay': 3.6283583803549155e-05, 'batch_size': 128}


=== Tuning TempBERT ===
Best trial: 6. Best value: 0.860641: 100%|███████████████████████████████████████████████████████| 10/10 [09:06<00:00, 54.60s/it]

[TempBERT] Miglior F1 medio in CV: 0.8606
[TempBERT] Migliori parametri: {'window': 8, 'num_attention_heads': 4, 'head_dim': 8, 'num_hidden_layers': 1, 'intermediate_size': 128, 'dropout': 0.31670187825521173, 'learning_rate': 0.0019375501864120402, 'weight_decay': 0.0025764174425233167, 'batch_size': 256}


=== Training finale XGBoost ===
[0]	validation_0-logloss:0.64228
[50]	validation_0-logloss:0.27560
[100]	validation_0-logloss:0.26761
[133]	validation_0-logloss:0.27237

=== XGBoost ===
Tempo di training: 0.3s
Accuracy: 0.7524
              precision    recall  f1-score   support

           0       0.69      0.91      0.79      1328
           1       0.87      0.59      0.70      1321

    accuracy                           0.75      2649
   macro avg       0.78      0.75      0.75      2649
weighted avg       0.78      0.75      0.75      2649

ft_psi                  0.205877
src_mac_entropy         0.173135
src_ip_entropy          0.150149
dst_mac_entropy         0.122909
c_used                  0.057179
bytes_entropy           0.034460
payload_size_entropy    0.026923
payload_entropy_mean    0.025988
pkts_entropy            0.025799
paf                     0.023712
dst_ip_entropy          0.023488
payload_entropy_cv      0.020262
durations_entropy       0.016255
cvb                     0.016157
mean_bytes              0.016071
cvd                     0.015818
mean_pkts               0.015597
cvp                     0.015406
mean_duration           0.014816
dtype: float32
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/xgboost_model.json


=== Training finale CatBoost ===
0:	learn: 0.7972774	test: 0.8782235	best: 0.8782235 (0)	total: 17.8ms	remaining: 17.8s
50:	learn: 0.8832967	test: 0.8904333	best: 0.8936809 (15)	total: 439ms	remaining: 8.17s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.8936809095
bestIteration = 15

Shrink model to first 16 iterations.

=== CatBoost ===
Tempo di training: 0.7s
Accuracy: 0.7293
              precision    recall  f1-score   support

           0       0.69      0.83      0.75      1328
           1       0.78      0.63      0.70      1321

    accuracy                           0.73      2649
   macro avg       0.74      0.73      0.73      2649
weighted avg       0.74      0.73      0.73      2649

              Feature Id  Importances
0         src_ip_entropy    19.943045
1        src_mac_entropy     9.199918
2   payload_entropy_mean     9.023443
3                    paf     8.708616
4                 ft_psi     6.534514
5                    cvp     6.030194
6                 c_used     5.673335
7          bytes_entropy     5.265250
8           pkts_entropy     5.031647
9   payload_size_entropy     4.778509
10    payload_entropy_cv     4.095619
11         mean_duration     2.844146
12             mean_pkts     2.812833
13     durations_entropy     2.158740
14                   cvb     2.031455
15       dst_mac_entropy     1.778447
16        dst_ip_entropy     1.725653
17            mean_bytes     1.401546
18                   cvd     0.963091
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/catboost_model.cbm


=== Training finale FeatBERT ===

=== FeatBERT ===
Tempo di training: 132.0s
Accuracy: 0.7678
              precision    recall  f1-score   support

           0       0.69      0.97      0.81      1328
           1       0.95      0.56      0.71      1321

    accuracy                           0.77      2649
   macro avg       0.82      0.77      0.76      2649
weighted avg       0.82      0.77      0.76      2649

src_mac_entropy         18.079960
src_ip_entropy          14.540359
pkts_entropy            11.138122
c_used                  10.452106
bytes_entropy            8.808247
dst_ip_entropy           6.791938
paf                      6.342705
cvp                      4.433410
payload_entropy_mean     3.458552
mean_duration            3.340904
payload_size_entropy     3.207163
dst_mac_entropy          2.414367
mean_pkts                1.799715
ft_psi                   1.654241
payload_entropy_cv       1.572813
mean_bytes               1.385870
cvb                      0.579528
cvd                      0.000000
durations_entropy        0.000000
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/feat_bert


=== Training finale TempBERT ===

=== TempBERT (W=8) ===
Tempo di training: 13.3s
Accuracy: 0.7980
              precision    recall  f1-score   support

           0       0.72      0.99      0.83      1328
           1       0.98      0.61      0.75      1321

    accuracy                           0.80      2649
   macro avg       0.85      0.80      0.79      2649
weighted avg       0.85      0.80      0.79      2649

src_ip_entropy          40.156406
src_mac_entropy         33.347014
mean_duration            6.960586
payload_entropy_mean     5.572305
bytes_entropy            3.301640
ft_psi                   2.915993
payload_size_entropy     2.870292
c_used                   2.749698
cvb                      1.151721
cvp                      0.974344
dst_mac_entropy          0.000000
paf                      0.000000
payload_entropy_cv       0.000000
durations_entropy        0.000000
pkts_entropy             0.000000
dst_ip_entropy           0.000000
cvd                      0.000000
mean_pkts                0.000000
mean_bytes               0.000000
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/temp_bert


=== Confronto finale ===
                accuracy  precision  recall     f1  train_time_s
model                                                           
XGBoost           0.7524     0.8699  0.5920 0.7045        0.2708
CatBoost          0.7293     0.7828  0.6329 0.6999        0.6803
FeatBERT          0.7678     0.9549  0.5609 0.7067      131.9604
TempBERT (W=8)    0.7980     0.9816  0.6064 0.7496       13.2829
```

# No data preprocessing

```txt

```

---
