.. spEnhance documentation master file, created by
   sphinx-quickstart on Sun Dec  7 16:00:41 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

spEnhance: A Computational Framework for Trustworthy Super-Resolution Enhancement in Spatial Omics
===================================================================================================================

spEnhance is a computational framework for trustworthy super-resolution enhancement in spatial omics data. 
It starts from spot-level spatial transcriptomics (ST) data, paired H&E image and single cell reference,
and reconstructs fine-grained, trustworthy super-resolution spatial gene expression. 

Installation
-------------
Install the latest version from https://github.com/dsong-lab/spEnhance

Documentation
--------------

.. toctree::
   :maxdepth: 2
   :caption: OVERVIEW

   about_spEnhance
   workflow

.. toctree::
   :maxdepth: 2
   :caption: TUTORIAL

   Image_Preprocessing
   Histology
   countsplit
   celltype
   genecluster
   imputation
