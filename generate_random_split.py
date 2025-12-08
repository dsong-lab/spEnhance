import numpy as np
import pandas as pd
import sys
# python generate_random_split.py $prefix $cnts_train_name $locs_train_name $cnts_val_name $locs_val_name
prefix = sys.argv[1] # 
cnts_train_name = sys.argv[2] # 'cnts_train_seed_1.csv'
locs_train_name = sys.argv[3] # 'locs_train_seed_1.csv'
cnts_val_name = sys.argv[4] # 'cnts_val_seed_1.csv'
locs_val_name = sys.argv[5] # 'locs_val_seed_1.csv'


cnts = pd.read_csv(f'{prefix}/cnts.csv')
locs = pd.read_csv(f'{prefix}/locs.csv')
np.random.seed(1)   # set seed
index_train = np.random.choice(cnts.shape[0], size=int(np.floor(cnts.shape[0] * 0.8)), replace=False)
index_val = np.setdiff1d(np.arange(cnts.shape[0]), index_train)

cnts_train = cnts.iloc[index_train]
cnts_train.to_csv(f'{prefix}/{cnts_train_name}', index=False)

locs_train = locs.iloc[index_train]
locs_train.to_csv(f'{prefix}/{locs_train_name}', index=False)

cnts_val = cnts.iloc[index_val]
cnts_val.to_csv(f'{prefix}/{cnts_val_name}', index=False)

locs_val = locs.iloc[index_val]
locs_val.to_csv(f'{prefix}/{locs_val_name}', index=False)
