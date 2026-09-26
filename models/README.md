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

---

## FTF

```txt
Numero di entry del dataset: 15948
label           0     1
scenario_id            
0             237   249
1             178   176
2             118   111
3             161   162
4             824   759
5             193   180
6             776   776
7             254   247
8              96    93
9             170   163
10            226   223
11            157   155
12            180   177
13            221   218
14            169   167
15            900   899
16            704   703
17            225   219
18            249   247
19           1047  1038
20             79    78
21            319   314
22            511   508
23             47    46

Deviazione standard tra scenari: 574.139278600484

Scenari nel train (tuning+fit): 16 -> [2, 3, 4, 5, 7, 9, 10, 12, 13, 15, 17, 19, 20, 21, 22, 23]
Scenari in validation (early stopping): 3 -> [1, 6, 14]
Scenari nel test finale: 5 -> [0, 8, 11, 16, 18]


=== Tuning CatBoost (Optuna + GroupKFold per scenario_id) ===
Best trial: 26. Best value: 0.835728: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [02:59<00:00,  3.58s/it]

[CatBoost] Miglior F1 medio in CV: 0.8357
[CatBoost] Migliori parametri: {'depth': 5, 'learning_rate': 0.13676799806194875, 'l2_leaf_reg': 4.714029494142138, 'rsm': 0.9345096954319009, 'bagging_temperature': 0.7514619264496079, 'random_strength': 2.123371598173719}


=== Tuning XGBoost (Optuna + GroupKFold per scenario_id) ===
Best trial: 44. Best value: 0.838081: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [02:23<00:00,  2.86s/it]

[XGBoost] Miglior F1 medio in CV: 0.8381
[XGBoost] Migliori parametri: {'max_depth': 5, 'learning_rate': 0.03151178591005771, 'reg_lambda': 4.244805768331018, 'colsample_bytree': 0.8151838362854762, 'subsample': 0.5434904271631307, 'min_child_weight': 2.6607524378461465}

[HF-BERT] Miglior F1 medio in CV: 0.7999
[HF-BERT] Migliori parametri: {'n_bins': 12, 'num_attention_heads': 8, 'head_dim': 24, 'num_hidden_layers': 2, 'intermediate_size': 512, 'learning_rate': 0.00017643967683381525, 'weight_decay': 9.833181933644887e-05, 'batch_size': 256}

=== Training finale CatBoost ===
0:	learn: 0.8095283	test: 0.6372458	best: 0.6372458 (0)	total: 2.64ms	remaining: 2.63s
50:	learn: 0.8506550	test: 0.7102003	best: 0.7156111 (13)	total: 153ms	remaining: 2.84s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.7156110975
bestIteration = 13

Shrink model to first 14 iterations.

=== CatBoost ===
Tempo di training: 0.2s
Accuracy: 0.7952
              precision    recall  f1-score   support

           0       0.71      0.98      0.83      1443
           1       0.97      0.61      0.75      1447

    accuracy                           0.80      2890
   macro avg       0.84      0.80      0.79      2890
weighted avg       0.84      0.80      0.79      2890

              Feature Id  Importances
0         src_ip_entropy    43.168788
1                 c_used     9.738455
2        src_mac_entropy     7.531213
3                 ft_psi     7.057569
4           pkts_entropy     6.875945
5                    paf     3.954127
6   payload_entropy_mean     3.773761
7        dst_mac_entropy     3.088358
8   payload_size_entropy     2.958828
9          bytes_entropy     2.836457
10     durations_entropy     2.761435
11        dst_ip_entropy     2.661496
12    payload_entropy_cv     1.633285
13         mean_duration     0.599689
14             mean_pkts     0.521695
15                   cvd     0.352009
16                   cvb     0.253159
17                   cvp     0.120059
18            mean_bytes     0.113672

=== XGBoost ===
Tempo di training: 0.4s
Accuracy: 0.8066
              precision    recall  f1-score   support

           0       0.73      0.97      0.83      1443
           1       0.96      0.64      0.77      1447

    accuracy                           0.81      2890
   macro avg       0.84      0.81      0.80      2890
weighted avg       0.84      0.81      0.80      2890

src_mac_entropy         0.287059
src_ip_entropy          0.147629
ft_psi                  0.090357
dst_mac_entropy         0.075732
c_used                  0.060251
payload_size_entropy    0.045534
paf                     0.044302
dst_ip_entropy          0.039265
payload_entropy_mean    0.031781
bytes_entropy           0.026643
payload_entropy_cv      0.023397
mean_duration           0.020163
durations_entropy       0.019006
pkts_entropy            0.018567
mean_bytes              0.016689
mean_pkts               0.014405
cvd                     0.014167
cvb                     0.012912
cvp                     0.012141
dtype: float32

=== HF-BERT (tokenizzato) ===
Tempo di training: 58.8s
Accuracy: 0.7671
              precision    recall  f1-score   support

           0       0.72      0.88      0.79      1443
           1       0.84      0.66      0.74      1447

    accuracy                           0.77      2890
   macro avg       0.78      0.77      0.76      2890
weighted avg       0.78      0.77      0.76      2890

=== Confronto finale ===
                       accuracy  precision  recall     f1  train_time_s
model                                                                  
CatBoost                 0.7952     0.9745  0.6068 0.7479        0.2397
XGBoost                  0.8066     0.9596  0.6406 0.7683        0.4405
HF-BERT (tokenizzato)    0.7671     0.8425  0.6579 0.7388       58.7990
```

## MFR

```txt
Numero di entry del dataset: 1174648

label             0       1
scenario_id                
0              4282   25717
1             12593   17703
2              5147    4933
3             13990    6150
4             28230   52279
5             18165    9302
6            123640   31716
7             15529   17979
8              5628    3536
9             12178    5584
10            13765   16349
11            17641    6175
12            16521    9151
13            17754   16656
14            12260   10655
15           108130   56930
16            69140   36692
17            23654   17677
18            30536   15223
19            49385  111124
20             4741    6157
21            19349   23481
22            27506   18906
23             3111    1699

Deviazione standard tra scenari: 48324.2402639208

Gruppi (scenario_id+group_id) nel train (tuning+fit): 5480
Gruppi in validation (early stopping): 968
Gruppi nel test finale: 1612


=== Tuning CatBoost (Optuna + GroupKFold) ===
Best trial: 20. Best value: 0.95329: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [1:25:43<00:00, 102.87s/it]

[CatBoost] Miglior F1 medio in CV: 0.9533
[CatBoost] Migliori parametri: {'depth': 10, 'learning_rate': 0.1932931411937336, 'l2_leaf_reg': 6.708345677163782, 'rsm': 0.5253041152433674, 'bagging_temperature': 2.0875443187939817, 'random_strength': 3.219106892035166}


=== Tuning XGBoost (Optuna + GroupKFold) ===
Best trial: 31. Best value: 0.945717: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [1:33:40<00:00, 112.42s/it]

[XGBoost] Miglior F1 medio in CV: 0.9457
[XGBoost] Migliori parametri: {'max_depth': 9, 'learning_rate': 0.010137894164845479, 'reg_lambda': 3.182504380756116, 'colsample_bytree': 0.7237071763380803, 'subsample': 0.9474920779627608, 'min_child_weight': 4.770297241115938}

[HF-BERT] Miglior F1 medio in CV: 0.9374
[HF-BERT] Migliori parametri: {'n_bins': 12, 'num_attention_heads': 8, 'head_dim': 24, 'num_hidden_layers': 2, 'intermediate_size': 512, 'learning_rate': 0.00017643967683381525, 'weight_decay': 9.833181933644887e-05, 'batch_size': 256}


=== Training finale CatBoost ===
0:	learn: 0.9252227	test: 0.9267474	best: 0.9267474 (0)	total: 90.2ms	remaining: 1m 30s
50:	learn: 0.9529227	test: 0.9523217	best: 0.9523734 (40)	total: 4.04s	remaining: 1m 15s
100:	learn: 0.9536964	test: 0.9528809	best: 0.9528887 (96)	total: 7.98s	remaining: 1m 11s
150:	learn: 0.9545511	test: 0.9530214	best: 0.9530800 (142)	total: 12.1s	remaining: 1m 7s
Stopped by overfitting detector  (50 iterations wait)

bestTest = 0.9530800442
bestIteration = 142

Shrink model to first 143 iterations.

=== CatBoost ===
Tempo di training: 15.7s
Accuracy: 0.9506
              precision    recall  f1-score   support

           0       0.99      0.93      0.96    130772
           1       0.91      0.98      0.94     94261

    accuracy                           0.95    225033
   macro avg       0.95      0.96      0.95    225033
weighted avg       0.95      0.95      0.95    225033

            Feature Id  Importances
0                  psi    36.655412
1           byte_count    17.738531
2         packet_count    17.488720
3  packets_per_seconds    14.845571
4    bytes_per_seconds     9.885260
5             duration     3.386505

=== XGBoost ===
Tempo di training: 33.7s
Accuracy: 0.9508
              precision    recall  f1-score   support

           0       0.99      0.93      0.96    130772
           1       0.91      0.98      0.94     94261

    accuracy                           0.95    225033
   macro avg       0.95      0.96      0.95    225033
weighted avg       0.95      0.95      0.95    225033

psi                    0.944014
packets_per_seconds    0.022209
packet_count           0.015017
bytes_per_seconds      0.008715
byte_count             0.007991
duration               0.002054
dtype: float32

=== HF-BERT (tokenizzato) ===
Tempo di training: 2017.3s
Accuracy: 0.9420
              precision    recall  f1-score   support

           0       0.98      0.92      0.95    130772
           1       0.89      0.98      0.93     94261

    accuracy                           0.94    225033
   macro avg       0.94      0.95      0.94    225033
weighted avg       0.95      0.94      0.94    225033

=== Confronto finale ===
                       accuracy  precision  recall     f1  train_time_s
model                                                                  
CatBoost                 0.9506     0.9059  0.9842 0.9434       15.7327
XGBoost                  0.9508     0.9064  0.9844 0.9437       33.6551
HF-BERT (tokenizzato)    0.9420     0.8927  0.9792 0.9339     2017.3008

```