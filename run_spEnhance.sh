#!/bin/bash
set -e
# example: bash run_stEnhance.sh $prefix true
prefix=$1  # e.g. data/demo/
validation=$(echo "${2:-false}" | tr '[:upper:]' '[:lower:]') # e.g. true or false, default is false if not provided
echo "validation = $validation"

device="cuda"  # "cuda" or "cpu"
pixel_size=0.5  # desired pixel size for the whole analysis
n_genes=1000  # number of most variable genes to impute

# preprocess histology image
echo $pixel_size > ${prefix}pixel-size.txt

python rescale.py ${prefix} --image

python preprocess.py ${prefix} --image

# Path to check
CHECK_FILE="checkpoints/vit256_small_dino.pth"
SCRIPT_TO_RUN="download_pretrained_vit.sh"

if [ ! -e "$CHECK_FILE" ]; then
    echo "File not found: $CHECK_FILE"
    echo "Running $SCRIPT_TO_RUN ..."
    bash "$SCRIPT_TO_RUN"
fi

# extract histology features
python extract_features_vit.py ${prefix} --device='cuda'
python extract_features_uni.py ${prefix} --login='HUGGINGFACE_TOKEN' # replace with your own login

python merge_feature.py ${prefix} --method='pca' # pca/nmf/ica

# auto detect tissue mask
python get_mask.py ${prefix}embeddings-hist-uni.pickle ${prefix}mask-small.png

# select most highly variable genes to predict
# If you have a user-defined list of genes, put it at `${prefix}gene-names.txt` and comment out the line below
python select_genes.py --n-top=${n_genes} "${prefix}cnts.csv" "${prefix}gene-names.txt"

# predict super-resolution gene expression
# rescale coordinates and spot radius
python rescale.py ${prefix} --locs --radius

python filter.py ${prefix}
if [ "$validation" = true ]; then
    echo "Generating validation data"

    cnts_train_name='cnts_train'
    cnts_val_name='cnts_val'
    seed=1
    # Make sure have R and countsplit package installed
    Rscript generate_count_split.R $prefix $cnts_train_name $cnts_val_name $seed 
    
    python rename_gene.py ${prefix}
    
    echo "Running imputation with validation setting"
    
    echo "Create embedding for train data"
    source cell_type_deconvolution.sh $prefix ${cnts_train_name}_seed_${seed}.csv locs.csv
    mv $prefix/embeddings-combined.pickle $prefix/embeddings-combined-train.pickle
    mv $prefix/predicted_celltype.png $prefix/predicted_celltype-train.png
    
    echo "Create embedding for validation data"
    source cell_type_deconvolution.sh $prefix ${cnts_val_name}_seed_${seed}.csv locs.csv
    mv $prefix/embeddings-combined.pickle $prefix/embeddings-combined-val.pickle
    mv $prefix/predicted_celltype.png $prefix/predicted_celltype-val.png

    echo "Create embedding for ALL data"
    source cell_type_deconvolution.sh $prefix cnts.csv locs.csv
    mv $prefix/embeddings-combined.pickle $prefix/embeddings-combined-all.pickle
    mv $prefix/predicted_celltype.png $prefix/predicted_celltype-all.png

    Rscript run_nnmf.R --path ${prefix} --noSignatures 15 --k 6 --ntop 50
    
    python impute_final_split.py --prefix ${prefix} \
        --cnts_train_name ${cnts_train_name}_seed_${seed}.csv \
        --locs_train_name locs.csv \
        --cnts_val_name ${cnts_val_name}_seed_${seed}.csv \
        --locs_val_name locs.csv \
        --epochs=400 --device='cuda' --n_states=5

    # Rank global performance and visualize local uncertainties
    python error_quant.py --prefix ${prefix} \
        --cnts_val_name ${cnts_val_name}_seed_${seed}.csv \
        --locs_val_name locs.csv
else
    # train gene expression prediction model and predict at super-resolution
    python impute_slide.py ${prefix} --epochs=400 --device='cuda' --n_states=5 # train model from scratch

    # python impute_slide.py ${prefix} --epochs=400 --device='cuda' --n_states=5 --load_saved # Load pretrained model
    # # If you want to retrain model, you need to delete the existing model:
    # rm -r ${prefix}states
    python impute_spot.py ${prefix}

    python impute_slide_aug.py ${prefix} --epochs=400 --device='cuda' --n_states=5
fi

# visualize imputed gene expression
python plot_imputed.py ${prefix} Prediction_val/cnts-super/

# segment image by gene features
python cluster.py --filter-size=8 --min-cluster-size=20 --n-clusters=10 --mask=${prefix}mask-small.png ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/

# # segment image without tissue mask
python cluster.py --filter-size=8 --min-cluster-size=20 ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unmasked/
# # segment image without spatial smoothing
# python cluster.py --mask=${prefix}mask-small.png ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unsmoothed/
# python cluster.py ${prefix}embeddings-gene.pickle ${prefix}clusters-gene/unsmoothed/unmasked/

# differential analysis by clusters
python aggregate_imputed.py ${prefix}
python reorganize_imputed.py ${prefix}
python differential.py ${prefix}

# visualize spot-level gene expression data
python plot_spots.py ${prefix}
# # cell type inference
# # see data/markers/cell-type-template.tsv for an example of a cell type reference panel
# python pixannot.py ${prefix} data/markers/cell-type.tsv ${prefix}markers/cell-type/
# cp -r ${prefix}markers/cell-type/threshold010/* ${prefix}markers/cell-type/
# python enrich.py ${prefix}clusters-gene/ ${prefix}markers/cell-type/

# # user-defined tissue structure signature scores
# # see data/markers/signature-score-template.tsv for an example of a signature score reference panel
# python marker_score.py ${prefix} data/markers/signature-score.txt ${prefix}markers/signature-score
