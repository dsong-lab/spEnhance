#!/usr/bin/env Rscript

# =============================
#   Gene clustering pipeline
#   Usage:
#   Rscript run_nnmf.R --path ~/spatial/ovart-5k/ --noSignatures 10 --k 5 --ntop 100
# =============================

# =============================
#   Gene clustering pipeline
#   Auto-install missing packages
# =============================

# -------------------------------
# Helper function: install if missing
# -------------------------------
install_if_missing <- function(pkgs) {
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message("Package '", pkg, "' not found. Installing...")
      if (pkg %in% rownames(installed.packages())) {
        suppressPackageStartupMessages(library(pkg, character.only = TRUE))
      } else {
        tryCatch({
          if (pkg %in% c("NNMF", "pheatmap", "optparse")) {
            install.packages(pkg, repos = "https://cloud.r-project.org")
          } else if (pkg %in% c("Matrix", "dplyr")) {
            install.packages(pkg, repos = "https://cloud.r-project.org")
          }
        }, error = function(e) {
          message("Failed to install ", pkg, ": ", e$message)
          quit(status = 1)
        })
      }
    }
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
}

# -------------------------------
# Packages required
# -------------------------------
pkgs <- c("NNMF", "Matrix", "dplyr", "pheatmap", "optparse", "ggplot2", "reshape2")
install_if_missing(pkgs)

suppressPackageStartupMessages({
  library(NNMF)
  library(Matrix)
  library(dplyr)
  library(pheatmap)
  library(optparse)
  library(ggplot2)
  library(reshape2)
})

# -------------------------------
# Parse command line arguments
# -------------------------------
option_list = list(
  make_option(c("--path"), type="character",
              help="Input folder path [default %default]"),
  make_option(c("--noSignatures"), type="integer", default=10,
              help="Number of signatures for NNMF [default %default]"),
  make_option(c("--k"), type="integer", default=5,
              help="Number of clusters for hierarchical clustering [default %default]"),
  make_option(c("--ntop"), type="integer", default=100,
              help="Number of top genes per signature [default %default]"),
  make_option(c("--outdir"), type="character", default=".",
              help="Output directory [default current dir]")
)

opt <- parse_args(OptionParser(option_list=option_list))

# -------------------------------
# Function to extract top features
# -------------------------------
topfeatures = function(signatures, feature_names, ntop = 10){
  if(ncol(signatures) != length(feature_names)){
    stop("The feature_names need to be the same length as the number of columns in signatures.\n")
  }
  dat = t(signatures)
  dat_new=NULL
  for(ii in 1:nrow(dat)){
    rr=dat[ii,]
    m1 = which.max(rr)
    m2 = which.max(rr[-m1])
    
    mm = rep(rr[m1], length(rr))
    mm[m1] = rr[m2]
    
    ns=rr*log(1 + rr/(mm+1e-10))
    dat_new=rbind(dat_new, ns)
  }
  
  weight_topgene = NULL
  for(topic in 1:ncol(dat)){
    idx = order(dat_new[,topic], decreasing = TRUE)
    weighting = feature_names[idx[1:ntop]]
    weight_topgene = rbind(weight_topgene,c(topic,weighting))
  }
  return(weight_topgene)
}

# -------------------------------
# Load data
# -------------------------------
message("Loading data...")
cnts <- read.csv(file.path(opt$path, "cnts.csv"), row.names = 1, check.names = FALSE)
locs <- read.csv(file.path(opt$path, "locs.csv"), row.names = 1)
gene_names <- as.vector(read.table(file.path(opt$path, "gene-names.txt")))

# Ensure genes order
cnts <- cnts[,gene_names[[1]]]

# Filter zero genes/spots
nonzero_cols <- which(Matrix::colSums(cnts) > 0)
cnts_filtered <- cnts[, nonzero_cols]
nonzero_rows <- which(Matrix::rowSums(cnts_filtered) > 0)
cnts_filtered <- cnts_filtered[nonzero_rows,]
locs <- locs[nonzero_rows,]

# -------------------------------
# Run NNMF
# -------------------------------
message("Running NNMF...")
res = nnmf(data = as.matrix(cnts_filtered),
           noSignatures = opt$noSignatures,
           location = as.matrix(locs), not_sc = F)

genes <- colnames(cnts_filtered)

# -------------------------------
# Rescaling and Normalization
# -------------------------------
features <- res$signatures
colnames(features) <- genes

mat <- as.matrix(features)

mat_new <- matrix(0, nrow = nrow(mat), ncol = ncol(mat))
rownames(mat_new) <- rownames(mat)
colnames(mat_new) <- colnames(mat)

# Reweighting
for (g in 1:ncol(mat)) {
  for (k in 1:nrow(mat)) {
    w_ki <- mat[k, g]
    # Find maximum on other signatures
    max_other <- max(mat[-k, g], na.rm = TRUE)
    # Avoid dividing by zero
    mat_new[k, g] <- w_ki * log(1 + w_ki / (max_other + 1e-9))
  }
}

df_new <- as.data.frame(mat_new)

# ------------------------------
# Clustering
# ------------------------------
# Transpose
gene_mat <- t(df_new)

# Scaling by gene
gene_mat_scaled <- scale(gene_mat)

# Similarity
dist_mat <- as.dist(1 - cor(t(gene_mat_scaled), method = "pearson"))

# Hierarchical clustering
hc <- hclust(dist_mat, method = "ward.D2")

# Cluster into 5 groups
k <- opt$k
clusters <- cutree(hc, k = k)

# Save results
gene_list <- split(names(clusters), clusters)

formatted <- paste0("[", 
                    paste(sapply(gene_list, function(x) {
                      paste0("[\"", paste(x, collapse="\",\""), "\"]")
                    }), collapse = ", "), 
                    "]")

writeLines(formatted, file.path(opt$path, "gene-names-group.txt"))

cat("Results saved !\n")
