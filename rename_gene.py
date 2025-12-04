import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    args = parser.parse_args()
    return args

args = get_args()
prefix = args.prefix

df_train = pd.read_csv(prefix + "cnts_train_seed_1.csv")

df_train.columns = [col.replace('.', '-') for col in df_train.columns]

df_train.to_csv(prefix + "cnts_train_seed_1.csv", index=False)

df_val = pd.read_csv(prefix + "cnts_val_seed_1.csv")

df_val.columns = [col.replace('.', '-') for col in df_val.columns]

df_val.to_csv(prefix + "cnts_val_seed_1.csv", index=False)