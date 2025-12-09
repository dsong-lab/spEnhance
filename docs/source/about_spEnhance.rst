About spEnhance
=======

The advent of spatial omics, especially spatial transcriptomics (ST), has enabled unprecedented insights into cellular heterogeneity and spatially regulated molecular programs. Within this spectrum of ST technologies, array-based platforms have become widely used for their transcriptome-wide coverage, but their spatial resolution remains limited. 

To overcome this barrier, numerous computational methods have been proposed to reconstruct high-resolution spatial gene expression from spot-level data by integrating histology; however, existing methods suffer from limited accuracy and reliability. Using only histological features, these methods are trained on a single dataset without a validation set, leading to overfitting and spurious gene expression patterns. 

Here we present spEnhance, a computational framework that reconstructs spatial gene expression with super-resolution from spot-level data while quantifying the confidence in its predictions. spEnhance achieves higher prediction accuracy by integrating both histology and single-cell RNA-seq (scRNA-seq) data, and employs a novel way to construct validation sets for trustworthy prediction. Beyond transcriptomics, spEnhance can also enhance other modalities, such as spatial alternative splicing and spatial chromatin accessibility. 

Comprehensive benchmarking across multiple tissues and platforms demonstrates that spEnhance achieves state-of-the-art accuracy, recovers fine-grained structure, and provides calibrated uncertainty estimates. Collectively, spEnhance establishes a generalizable and trustworthy framework for enhancing diverse spatial omics data, paving the way for more integrative and reliable analyses of spatial gene regulation.

