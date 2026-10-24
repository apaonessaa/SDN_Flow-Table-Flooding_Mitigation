# Stats

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
Best trial: 15. Best value: 0.836815: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [02:36<00:00,  7.83s/it]

[XGBoost] Miglior F1 medio in CV: 0.8368
[XGBoost] Migliori parametri: {'max_depth': 8, 'learning_rate': 5.907390928021829e-05, 'reg_lambda': 2.1090109114538387, 'colsample_bytree': 0.5180181096774437, 'subsample': 0.6360500990321296, 'min_child_weight': 0.0024244206021583046}


=== Tuning CatBoost ===
Best trial: 13. Best value: 0.83158: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:54<00:00,  2.71s/it]

[CatBoost] Miglior F1 medio in CV: 0.8316
[CatBoost] Migliori parametri: {'depth': 9, 'learning_rate': 0.00014014640289440795, 'l2_leaf_reg': 2.244230710868361, 'rsm': 0.8897860844614501, 'bagging_temperature': 0.4000410597964097}


=== Tuning FeatBERT ===
Best trial: 19. Best value: 0.843023: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [1:05:58<00:00, 197.92s/it]

[FeatBERT] Miglior F1 medio in CV: 0.8430
[FeatBERT] Migliori parametri: {'num_attention_heads': 8, 'head_dim': 24, 'num_hidden_layers': 4, 'intermediate_size': 512, 'dropout': 0.24812944757963573, 'learning_rate': 0.0005659194095571004, 'weight_decay': 4.919535479856801e-05, 'batch_size': 256}


=== Tuning TempBERT ===
Best trial: 12. Best value: 0.860228: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [25:40<00:00, 77.03s/it]

[TempBERT] Miglior F1 medio in CV: 0.8602
[TempBERT] Migliori parametri: {'window': 8, 'num_attention_heads': 4, 'head_dim': 32, 'num_hidden_layers': 3, 'intermediate_size': 512, 'dropout': 0.41080962316399994, 'learning_rate': 0.0009590390630689236, 'weight_decay': 2.4958408699460776e-06, 'batch_size': 256}


=== Training finale XGBoost ===
[0]	validation_0-logloss:0.69312
[50]	validation_0-logloss:0.69157
[100]	validation_0-logloss:0.68999
[150]	validation_0-logloss:0.68845
[200]	validation_0-logloss:0.68695
[250]	validation_0-logloss:0.68554
[300]	validation_0-logloss:0.68401
[350]	validation_0-logloss:0.68252
[400]	validation_0-logloss:0.68101
[450]	validation_0-logloss:0.67952
[500]	validation_0-logloss:0.67802
[550]	validation_0-logloss:0.67653
[600]	validation_0-logloss:0.67512
[650]	validation_0-logloss:0.67374
[700]	validation_0-logloss:0.67234
[750]	validation_0-logloss:0.67088
[800]	validation_0-logloss:0.66941
[850]	validation_0-logloss:0.66800
[900]	validation_0-logloss:0.66655
[950]	validation_0-logloss:0.66515
[999]	validation_0-logloss:0.66379

=== XGBoost ===
Tempo di training: 2.3s
Accuracy: 0.7614
              precision    recall  f1-score   support

           0       0.70      0.93      0.80      1328
           1       0.89      0.60      0.71      1321

    accuracy                           0.76      2649
   macro avg       0.79      0.76      0.75      2649
weighted avg       0.79      0.76      0.75      2649

src_mac_entropy         0.035789
src_ip_entropy          0.010710
c_used                  0.008457
payload_entropy_mean    0.005112
paf                     0.004229
bytes_entropy           0.003591
ft_psi                  0.003561
pkts_entropy            0.002120
payload_size_entropy    0.002104
mean_pkts               0.001587
dst_mac_entropy         0.000478
mean_bytes              0.000161
payload_entropy_cv     -0.000143
mean_duration          -0.000219
dst_ip_entropy         -0.000266
cvp                    -0.000507
cvd                    -0.000519
cvb                    -0.001128
durations_entropy      -0.003920
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/xgboost_model.json


=== Training finale CatBoost ===
0:	learn: 0.8075951	test: 0.8620572	best: 0.8620572 (0)	total: 19.4ms	remaining: 19.4s
50:	learn: 0.8115224	test: 0.8746439	best: 0.8770177 (2)	total: 760ms	remaining: 14.1s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.8770177366
bestIteration = 2

Shrink model to first 3 iterations.

=== CatBoost ===
Tempo di training: 0.9s
Accuracy: 0.7244
              precision    recall  f1-score   support

           0       0.67      0.87      0.76      1328
           1       0.82      0.58      0.68      1321

    accuracy                           0.72      2649
   macro avg       0.75      0.72      0.72      2649
weighted avg       0.75      0.72      0.72      2649

src_ip_entropy          0.104808
c_used                  0.028956
src_mac_entropy         0.021255
pkts_entropy            0.012745
bytes_entropy           0.002610
ft_psi                  0.002192
payload_size_entropy    0.000604
mean_bytes              0.000000
cvp                     0.000000
mean_pkts               0.000000
cvd                     0.000000
payload_entropy_mean    0.000000
payload_entropy_cv      0.000000
paf                     0.000000
cvb                     0.000000
dst_ip_entropy         -0.000313
dst_mac_entropy        -0.000870
mean_duration          -0.003396
durations_entropy      -0.007157
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/catboost_model.cbm


=== Training finale FeatBERT ===

=== FeatBERT ===
Tempo di training: 211.5s
Accuracy: 0.7444
              precision    recall  f1-score   support

           0       0.69      0.87      0.77      1328
           1       0.83      0.61      0.71      1321

    accuracy                           0.74      2649
   macro avg       0.76      0.74      0.74      2649
weighted avg       0.76      0.74      0.74      2649

src_mac_entropy         0.032856
c_used                  0.028976
src_ip_entropy          0.022097
pkts_entropy            0.021557
bytes_entropy           0.012374
dst_ip_entropy          0.006929
mean_duration           0.005752
payload_size_entropy    0.005532
payload_entropy_mean    0.004822
cvp                     0.004217
dst_mac_entropy         0.004161
mean_pkts               0.003619
ft_psi                  0.002562
mean_bytes              0.001002
payload_entropy_cv      0.000853
cvd                    -0.000081
cvb                    -0.001431
paf                    -0.001829
durations_entropy      -0.011141
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/feat_bert


=== Training finale TempBERT ===

=== TempBERT (W=8) ===
Tempo di training: 51.8s
Accuracy: 0.8048
              precision    recall  f1-score   support

           0       0.72      0.99      0.84      1328
           1       0.99      0.62      0.76      1321

    accuracy                           0.80      2649
   macro avg       0.86      0.80      0.80      2649
weighted avg       0.86      0.80      0.80      2649

src_mac_entropy         0.113685
src_ip_entropy          0.071880
payload_entropy_mean    0.016589
mean_duration           0.015583
c_used                  0.007882
pkts_entropy            0.006801
payload_size_entropy    0.004506
cvd                     0.004503
ft_psi                  0.003708
cvp                     0.003462
bytes_entropy           0.003405
paf                    -0.000103
mean_pkts              -0.000545
cvb                    -0.000687
mean_bytes             -0.003504
payload_entropy_cv     -0.006498
durations_entropy      -0.013888
dst_ip_entropy         -0.014803
dst_mac_entropy        -0.016763
dtype: float64
Model path: /home/ubuntu/Desktop/SDN_Flow-Table-Flooding_Mitigation/models/ftf_detector/temp_bert


=== Confronto finale ===
                accuracy  precision  recall     f1  train_time_s
model                                                           
XGBoost           0.7614     0.8901  0.5950 0.7132        2.3249
CatBoost          0.7244     0.8167  0.5768 0.6761        0.8542
FeatBERT          0.7444     0.8292  0.6139 0.7055      211.5080
TempBERT (W=8)    0.8048     0.9891  0.6154 0.7587       51.8412

```