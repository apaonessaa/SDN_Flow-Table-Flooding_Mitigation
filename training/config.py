"""
    Configuration model file
"""
# Tuning + Training
N_TRIALS = 3
N_SPLITS = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.15
CV_EARLY_STOPPING_ROUNDS = 50
FINAL_EARLY_STOPPING_ROUNDS = 50
RANDOM_SEED = 42

# Optuna Catboost & XGBoost
# Revisiting Deep Learning Models for Tabular Data (Table 16-17)
DEPTH = (3, 10)
BOOST_LEARNING_RATE = (1e-5, 1)
ITERATIONS = 1000
COLSAMPLE = (0.5, 1.0)

#XGBoost
LAMBDA = (1e-8, 1e2)
SUBSAMPLE = (0.5, 1.0)
MIN_CHILD_WEIGHT = (1e-8, 1e5)

#CatBoost
L2_REG = (1.0, 10.0)
BAGGING_TEMPERATURE = (0.0, 1.0)

# Optuna Tuning BERT
# Revisiting Deep Learning Models for Tabular Data (Table 13)
BERT_CV_EPOCHS = 30
BERT_CV_PATIENCE = 5
NUM_ATTENTION_HEADS = [2,4,8]
HEAD_DIM = [8,16,24,32]
NUM_HIDDEN_LAYERS = (1,6)
INTERMEDIATE_SIZE = [64,128,256,512]
LEARNING_RATE = (1e-5,1e-3)
WEIGHT_DECAY = (1e-6,1e-3)
DROPOUT = (0.0,0.5)
BATCH_SIZE = [256,512,1024]

# Optuna Training+Validation BERT
BERT_FINAL_EPOCHS = 50
BERT_FINAL_PATIENCE = 10

# TempBERT
WINDOWS = [3,5,8,12]
