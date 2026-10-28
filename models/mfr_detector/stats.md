
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

```