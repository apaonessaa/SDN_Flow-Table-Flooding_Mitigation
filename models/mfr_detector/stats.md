
```txt
Numero di entry del dataset: 486794
label            0      1
scenario_id              
0             1444  26077
1             6155  19361
2             8138   2896
3             9839  13020
4            21055  94231
5             9234   4407
6            38565  41190
7             4893  14520
8             2373   3457
9             4833   7969
10            4609  11562
11            3098   2890
12            4716   8421
13            6624  15514
14            2002   7042
15           32046  54613

Deviazione standard tra scenari: 32869.82265619941
[!] Stratificazione per 'attack_profile' non applicabile (The test_size = 2 should be greater or equal to the number of classes = 3); eseguo uno split non stratificato.


=== Tuning FeatBERT ===
Best trial: 18. Best value: 0.976591: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [19:21:08<00:00, 3483.42s/it]

[FeatBERT] Miglior F1 medio in CV: 0.9766
[FeatBERT] Migliori parametri: {'num_attention_heads': 4, 'head_dim': 8, 'num_hidden_layers': 6, 'intermediate_size': 128, 'dropout': 0.36973162825872385, 'learning_rate': 0.0001811285009417781, 'weight_decay': 3.233211565346471e-06, 'batch_size': 1024}


=== Tuning TempBERT ===
Best trial: 18. Best value: 0.994649: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [25:31:40<00:00, 4595.03s/it]

[TempBERT] Miglior F1 medio in CV: 0.9946
[TempBERT] Migliori parametri: {'window': 12, 'num_attention_heads': 4, 'head_dim': 32, 'num_hidden_layers': 4, 'intermediate_size': 64, 'dropout': 0.11005184344936608, 'learning_rate': 0.0001539629976525771, 'weight_decay': 0.00024274385186805834, 'batch_size': 256}


=== Training finale FeatBERT ===

=== FeatBERT ===
Tempo di training: 1538.3s
Accuracy: 0.8759
              precision    recall  f1-score   support

           0       0.79      0.96      0.87     48274
           1       0.97      0.81      0.88     62684

    accuracy                           0.88    110958
   macro avg       0.88      0.89      0.88    110958
weighted avg       0.89      0.88      0.88    110958

packet_count           0.331364
byte_count             0.072488
duration               0.068477
packets_per_seconds    0.067111
psi                    0.008947
bytes_per_seconds     -0.009165
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/mfr_detector/feat_bert

=== TempBERT (W=12) ===
Tempo di training: 4676.6s
Accuracy: 0.9958
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     48274
           1       1.00      1.00      1.00     62684

    accuracy                           1.00    110958
   macro avg       1.00      1.00      1.00    110958
weighted avg       1.00      1.00      1.00    110958

packet_count           0.546585
packets_per_seconds    0.341660
bytes_per_seconds      0.299864
byte_count             0.125454
duration               0.014291
psi                    0.005913
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/mfr_detector/temp_bert


=== Tuning XGBoost ===
Best trial: 12. Best value: 0.97515: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [20:20<00:00, 61.02s/it]

[XGBoost] Miglior F1 medio in CV: 0.9751
[XGBoost] Migliori parametri: {'max_depth': 10, 'learning_rate': 0.0001616883334680379, 'reg_lambda': 1.0429855041098153, 'colsample_bytree': 0.6621885229657863, 'subsample': 0.7135561528166658, 'min_child_weight': 7.871071394097009e-05}


=== Tuning CatBoost ===
Best trial: 17. Best value: 0.946064: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [04:02<00:00, 12.14s/it]

[CatBoost] Miglior F1 medio in CV: 0.9461
[CatBoost] Migliori parametri: {'depth': 8, 'learning_rate': 0.00017917911413641767, 'l2_leaf_reg': 2.5497807206321017, 'rsm': 0.9270627421637063, 'bagging_temperature': 0.8796139267265123}


=== Training finale XGBoost ===
[0]	validation_0-logloss:0.69303
[50]	validation_0-logloss:0.68672
[100]	validation_0-logloss:0.68074
[150]	validation_0-logloss:0.67499
[200]	validation_0-logloss:0.66920
[250]	validation_0-logloss:0.66333
[300]	validation_0-logloss:0.65747
[350]	validation_0-logloss:0.65184
[400]	validation_0-logloss:0.64650
[450]	validation_0-logloss:0.64091
[500]	validation_0-logloss:0.63556
[550]	validation_0-logloss:0.63035
[600]	validation_0-logloss:0.62503
[650]	validation_0-logloss:0.61996
[700]	validation_0-logloss:0.61476
[750]	validation_0-logloss:0.60987
[800]	validation_0-logloss:0.60490
[850]	validation_0-logloss:0.59978
[900]	validation_0-logloss:0.59493
[950]	validation_0-logloss:0.59035
[999]	validation_0-logloss:0.58572

=== XGBoost ===
Tempo di training: 14.7s
Accuracy: 0.9633
              precision    recall  f1-score   support

           0       0.96      0.96      0.96     48274
           1       0.97      0.97      0.97     62684

    accuracy                           0.96    110958
   macro avg       0.96      0.96      0.96    110958
weighted avg       0.96      0.96      0.96    110958

packet_count           0.308658
packets_per_seconds    0.246504
bytes_per_seconds      0.085990
byte_count             0.082456
psi                    0.008733
duration               0.002466
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/mfr_detector/xgboost_model.json


=== Training finale CatBoost ===
0:	learn: 0.9440865	test: 0.9585914	best: 0.9585914 (0)	total: 35.3ms	remaining: 35.3s
50:	learn: 0.9427210	test: 0.9545255	best: 0.9589787 (2)	total: 1.87s	remaining: 34.8s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.9589786519
bestIteration = 2

Shrink model to first 3 iterations.

=== CatBoost ===
Tempo di training: 2.1s
Accuracy: 0.8439
              precision    recall  f1-score   support

           0       0.75      0.96      0.84     48274
           1       0.96      0.76      0.85     62684

    accuracy                           0.84    110958
   macro avg       0.85      0.86      0.84    110958
weighted avg       0.87      0.84      0.84    110958

packet_count           0.266997
packets_per_seconds    0.159722
byte_count             0.103603
duration               0.079503
psi                    0.007333
bytes_per_seconds     -0.000496
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/mfr_detector/catboost_model.cbm


=== Confronto finale ===
                 accuracy  precision  recall     f1  train_time_s
model                                                            
FeatBERT           0.8759     0.9700  0.8100 0.8800     1538.3000
TempBERT (W=12)    0.9958     0.9961  0.9964 0.9962     4676.5602
XGBoost            0.9633     0.9657  0.9696 0.9676       14.6722
CatBoost           0.8439     0.9572  0.7577 0.8458        2.1280

```