# spstEnhance

+ Refer to `run_stEhance.sh` for how to run the pipeline.
+ Put single cell reference in the same directory as `cnts.csv`, `locs-raw.csv` and other input files. Name your single cell reference, which is expected as a seurat object, as `sc_reference.RDS` and save your cell type annotation as `celltype`.
+ When doing cell type annotation in the single cell data, try to keep the total number of cell types around 8 or so, such that the proportion of each type of cell wouldn't be too low, which is beneficial for trianing process.
+ If sc_reference is available, it seems unnecessary to do spot imputation. Even if the reference is not paired with spatial transcriptomics data, it still improve the performance to a certain degree, as long as the data has been well annotated.
+ However, if no single reference available, even public ones, then we would stick to the old pipeline.

## Update on Sept 28, 2025
+ rename single-cell reference as `sc_reference.RDS` and put the active annotation into celltype.
+ Run `run_nnmf.R` before running imputation and make sure the file `gene-names-group.txt` exist in the same directory as other input files.
