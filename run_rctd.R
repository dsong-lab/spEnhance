#!/usr/bin/env Rscript

# ========== Install dependencies ==========
install_if_missing <- function(pkgs) {
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message("Installing missing package: ", pkg)
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
  }
}

# Necessary CRAN packages
cran_pkgs <- c("ggplot2", "Matrix")
install_if_missing(cran_pkgs)

# Bioconductor 
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}
bioc_pkgs <- c("SpatialExperiment", "SummarizedExperiment", "spacexr")
for (pkg in bioc_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing missing Bioconductor package: ", pkg)
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

# ========== Load Dependencies ==========
suppressMessages({
  library(ggplot2)
  library(SpatialExperiment)
  library(SummarizedExperiment)
  library(Seurat)
  library(spacexr)
  library(Matrix)
})

# ========== Parameters ==========
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript run_rctd.R <seurat_ref.rds> <cnts.csv> <locs.csv> <outdir> <num_cores> [seed]")
}

seurat_file <- args[1]  # annotated scRNA-seq (Seurat RDS)
cnts_file   <- args[2]  # spatial counts
locs_file   <- args[3]  # spatial coords
outdir      <- args[4]  # output folder
num_cores   <- args[5]  # number of cores used
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)
if (length(args) >= 6) {
  seed <- suppressWarnings(as.integer(args[6]))
  if (!is.na(seed)) {
    set.seed(seed)
  }
}

# ========== Load single-cell reference ==========
print("Loading single-cell reference...")
seurat_ref <- readRDS(seurat_file)

counts_matrix <- GetAssayData(seurat_ref, layer = "counts")  
cell_types <- data.frame(cell_type = seurat_ref$celltype)
rownames(cell_types) <- colnames(seurat_ref)  

reference_se <- SummarizedExperiment(
  assays = list(counts = counts_matrix),
  colData = cell_types
)
print("Reference loaded!")
# ========== Load spatial transcriptomics data ==========
print("Loading Spatial Trascriptomics Data...")
cnts <- read.csv(cnts_file, row.names = 1, check.names = FALSE)
locs <- read.csv(locs_file, row.names = 1)

var_genes <- apply(cnts, 2, var)
cnts <- cnts[, var_genes > 0, drop = FALSE]
cnts <- t(cnts)
cnts <- cnts[complete.cases(cnts), ]
stopifnot(all(rownames(locs) == colnames(cnts)))

spatial_spe <- SpatialExperiment(
  assay = as.matrix(cnts),
  spatialCoords = as.matrix(locs)
)
print("ST data loaded!")
# ========== Run RCTD ==========
print("Running RCTD...")
rctd_data <- createRctd(spatial_spe, reference_se)
results_spe <- runRctd(rctd_data, rctd_mode = "multi", max_cores = as.integer(num_cores))
print("Finished!")
# ========== Save results ==========
weights <- assay(results_spe, "weights")
weights <- t(weights)

common_spots <- intersect(rownames(weights), rownames(locs))
weights_filt <- weights[common_spots, , drop = FALSE]
locs_filt    <- locs[common_spots, , drop = FALSE]
stopifnot(all(rownames(weights_filt) == rownames(locs_filt)))

write.csv(weights_filt, file = file.path(outdir, "proportion_celltype.csv"), quote = FALSE)
write.csv(locs_filt,    file = file.path(outdir, "locs_celltype.csv"), quote = FALSE)

# ========== Build reference matrix ==========
celltype <- seurat_ref$celltype
keep_cells <- !is.na(celltype) & celltype != ""
expr_mat <- GetAssayData(seurat_ref, layer = "counts")[, keep_cells]
celltype <- celltype[keep_cells]

expr <- as.matrix(expr_mat)
ref_matrix <- sapply(unique(celltype), function(ct) {
  rowMeans(expr_mat[, celltype == ct, drop = FALSE])
})
ref_matrix <- t(ref_matrix)

cnts_genes <- rownames(cnts)
ref_matrix_filtered <- ref_matrix[, colnames(ref_matrix) %in% cnts_genes, drop = FALSE]

write.csv(ref_matrix_filtered, file = file.path(outdir, "reference.csv"), quote = FALSE)

cat("RCTD analysis finished. Results saved in:", outdir, "\n")
