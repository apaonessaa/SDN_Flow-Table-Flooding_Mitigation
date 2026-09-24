# README

Scenarios 0-15

## FTF

```txt
Numero di entry del dataset: 9614
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

Scenari nel train (tuning+fit): 10 -> [2, 3, 4, 6, 7, 8, 9, 10, 11, 15]
Scenari in validation (early stopping): 2 -> [12, 13]
Scenari nel test finale: 4 -> [0, 1, 5, 14]

[CatBoost] Miglior F1 medio in CV: 0.7547
[CatBoost] Migliori parametri: {'depth': 5, 'learning_rate': 0.038364415605180335, 'l2_leaf_reg': 8.489550140301192, 'rsm': 0.795886142463212, 'bagging_temperature': 0.6900364393712313, 'random_strength': 2.9076836872018506}

[XGBoost] Miglior F1 medio in CV: 0.7508
[XGBoost] Migliori parametri: {'max_depth': 5, 'learning_rate': 0.010788306146610917, 'reg_lambda': 11.228639409006483, 'colsample_bytree': 0.9145045841310313, 'subsample': 0.6589889903222934, 'min_child_weight': 8.567055773908631}

[HF-BERT] Miglior F1 medio in CV: 0.7435
[HF-BERT] Migliori parametri: {'n_bins': 12, 'num_attention_heads': 8, 'head_dim': 24, 'num_hidden_layers': 2, 'intermediate_size': 512, 'learning_rate': 0.00017643967683381525, 'weight_decay': 9.833181933644887e-05, 'batch_size': 256}

=== Training finale CatBoost ===
0:	learn: 0.7221356	test: 0.8666030	best: 0.8666030 (0)	total: 8.06ms	remaining: 8.05s
50:	learn: 0.7540717	test: 0.8875064	best: 0.8898671 (40)	total: 129ms	remaining: 2.4s
100:	learn: 0.7611161	test: 0.8843329	best: 0.8900084 (85)	total: 247ms	remaining: 2.2s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.8900084445
bestIteration = 85

Shrink model to first 86 iterations.

=== CatBoost ===
Tempo di training: 0.4s
Accuracy: 0.8941
              precision    recall  f1-score   support

           0       0.88      0.91      0.90       777
           1       0.91      0.87      0.89       772

    accuracy                           0.89      1549
   macro avg       0.89      0.89      0.89      1549
weighted avg       0.89      0.89      0.89      1549

              Feature Id  Importances
0         src_ip_entropy    32.796807
1        src_mac_entropy    19.489442
2                 c_used     8.542589
3                    paf     8.322378
4                 ft_psi     3.916247
5   payload_entropy_mean     3.900573
6          bytes_entropy     3.221270
7           pkts_entropy     2.831207
8   payload_size_entropy     2.694462
9        dst_mac_entropy     2.440135
10        dst_ip_entropy     2.402654
11    payload_entropy_cv     1.615365
12     durations_entropy     1.588549
13             mean_pkts     1.557963
14                   cvp     1.150599
15         mean_duration     1.062175
16                   cvd     0.940167
17                   cvb     0.779392
18            mean_bytes     0.748023

=== XGBoost ===
Tempo di training: 1.5s
Accuracy: 0.9109
              precision    recall  f1-score   support

           0       0.90      0.93      0.91       777
           1       0.93      0.89      0.91       772

    accuracy                           0.91      1549
   macro avg       0.91      0.91      0.91      1549
weighted avg       0.91      0.91      0.91      1549

src_mac_entropy         0.326104
paf                     0.121536
ft_psi                  0.089753
src_ip_entropy          0.078062
c_used                  0.046393
dst_mac_entropy         0.032820
payload_entropy_mean    0.029341
dst_ip_entropy          0.029067
bytes_entropy           0.027681
payload_size_entropy    0.026838
pkts_entropy            0.024523
payload_entropy_cv      0.024018
durations_entropy       0.022630
mean_duration           0.021225
mean_pkts               0.020991
cvp                     0.020229
cvb                     0.020149
mean_bytes              0.020124
cvd                     0.018517
dtype: float32

=== HF-BERT (tokenizzato) ===
Tempo di training: 37.9s
Accuracy: 0.8722
              precision    recall  f1-score   support

           0       0.88      0.86      0.87       777
           1       0.87      0.88      0.87       772

    accuracy                           0.87      1549
   macro avg       0.87      0.87      0.87      1549
weighted avg       0.87      0.87      0.87      1549


=== Confronto finale ===
                       accuracy  precision  recall     f1  train_time_s
model                                                                  
CatBoost                 0.8941     0.9097  0.8744 0.8917        0.3720
XGBoost                  0.9109     0.9261  0.8925 0.9090        1.4999
HF-BERT (tokenizzato)    0.8722     0.8651  0.8808 0.8729       37.9114

```

## MFR

```txt
Numero di entry del dataset: 716267

label             0      1
scenario_id               
0              4282  25717
1             12593  17703
2              5147   4933
3             13990   6150
4             28230  52279
5             18165   9302
6            123640  31716
7             15529  17979
8              5628   3536
9             12178   5584
10            13765  16349
11            17641   6175
12            16521   9151
13            17754  16656
14            12260  10655
15           108130  56930

Deviazione standard tra scenari: 47777.926562378154

Gruppi (scenario_id+group_id) nel train (tuning+fit): 3317
Gruppi in validation (early stopping): 586
Gruppi nel test finale: 976

[CatBoost] Miglior F1 medio in CV: 0.9501
[CatBoost] Migliori parametri: {'depth': 5, 'learning_rate': 0.18030684085310186, 'l2_leaf_reg': 5.584868226999494, 'rsm': 0.7907590257331211, 'bagging_temperature': 1.1392120049598298, 'random_strength': 2.0576623564347645}

[XGBoost] Miglior F1 medio in CV: 0.9334
[XGBoost] Migliori parametri: {'max_depth': 10, 'learning_rate': 0.04248157134330249, 'reg_lambda': 1.804727100992134, 'colsample_bytree': 0.9985393243649586, 'subsample': 0.8767712224214339, 'min_child_weight': 6.247368928500291}

[HF-BERT] Miglior F1 medio in CV: 0.9241
[HF-BERT] Migliori parametri: {'n_bins': 12, 'num_attention_heads': 8, 'head_dim': 24, 'num_hidden_layers': 2, 'intermediate_size': 512, 'learning_rate': 0.00017643967683381525, 'weight_decay': 9.833181933644887e-05, 'batch_size': 256}


=== Training finale CatBoost ===
0:	learn: 0.8842511	test: 0.8878366	best: 0.8878366 (0)	total: 40.1ms	remaining: 40.1s
50:	learn: 0.9486408	test: 0.9494777	best: 0.9503358 (26)	total: 2.04s	remaining: 37.9s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.950335772
bestIteration = 26

Shrink model to first 27 iterations.

=== CatBoost ===
Tempo di training: 3.2s
Accuracy: 0.9375
              precision    recall  f1-score   support

           0       0.98      0.91      0.95     84554
           1       0.88      0.98      0.92     53531

    accuracy                           0.94    138085
   macro avg       0.93      0.94      0.94    138085
weighted avg       0.94      0.94      0.94    138085

            Feature Id  Importances
0                  psi    37.506325
1         packet_count    26.412161
2           byte_count    22.481484
3  packets_per_seconds     6.866647
4    bytes_per_seconds     5.835068
5             duration     0.898314

=== XGBoost ===
Tempo di training: 6.4s
Accuracy: 0.9409
              precision    recall  f1-score   support

           0       0.99      0.92      0.95     84554
           1       0.88      0.98      0.93     53531

    accuracy                           0.94    138085
   macro avg       0.93      0.95      0.94    138085
weighted avg       0.95      0.94      0.94    138085

psi                    0.934118
packets_per_seconds    0.026301
packet_count           0.016532
bytes_per_seconds      0.014410
byte_count             0.007322
duration               0.001317
dtype: float32

=== HF-BERT (tokenizzato) ===
Tempo di training: 1567.3s
Accuracy: 0.9313
              precision    recall  f1-score   support

           0       0.98      0.91      0.94     84554
           1       0.87      0.97      0.92     53531

    accuracy                           0.93    138085
   macro avg       0.92      0.94      0.93    138085
weighted avg       0.94      0.93      0.93    138085


=== Confronto finale ===
                       accuracy  precision  recall     f1  train_time_s
model                                                                  
CatBoost                 0.9375     0.8768  0.9760 0.9238        3.1716
XGBoost                  0.9409     0.8821  0.9783 0.9277        6.3565
HF-BERT (tokenizzato)    0.9313     0.8666  0.9724 0.9165     1567.2885

```