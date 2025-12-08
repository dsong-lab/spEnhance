#!/bin/bash
prefix=$1
cnts_name=$2
locs_name=$3
# source ~/.bashrc
# source /home/FCAM/dosong/miniconda3/etc/profile.d/conda.sh
micromamba activate pisces-rabbit
Rscript run_rctd.R ${prefix}sc_reference.RDS ${prefix}${cnts_name} ${prefix}${locs_name} ${prefix} 4 
conda activate stEnhance
python select_genes.py --n-top=600 ${prefix}"proportion_celltype.csv" ${prefix}"cell-type-names.txt"

python impute_slide_celltype.py ${prefix} --epochs=100 --device='cuda' --n_states=5

python assign_reference.py ${prefix} --mode='combined' --normalize='gene-zscore' --dim=256

python plot_predicted_celltype.py ${prefix}
