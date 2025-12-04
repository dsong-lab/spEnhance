from histological_feature_map import main as his_ex
from pos_feature_map import main as pos_ex
from RGB_feature_map import main as rgb_ex
import pickle
import argparse

# ====VPN===
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

# ====Feature extraction with UNI====
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--login', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    login = args.login  # Please change to your applied UNI login
    img_emb = his_ex(args.prefix, login)
    print('Shape of img feature: ' + str(img_emb[0].shape))
    rgb_emb = rgb_ex(args.prefix)
    print('Shape of rgb feature: ' + str(rgb_emb[0].shape))
    pos_emb = pos_ex(args.prefix, img_emb)
    print('Shape of pos feature: ' + str(pos_emb[0].shape))
    
    embs = dict(his=img_emb, rgb=rgb_emb, pos=pos_emb)
    with open(args.prefix + 'embeddings-hist-uni.pickle', 'wb') as file:
        pickle.dump(embs, file)
    print('embeddings-hist-uni saved!')

if __name__ == '__main__':
    main()
