spEnhance Workflow
===================
spEnhance reconstructs high-resolution spatial expression from spot-level spatial transcriptomics or other types of omics techniques by leveraging H&E-stained histology images and single-cell reference information. 

In this tutorial, we use spatial transcriptomics as the primary illustrative example.

We use a human ovarian cancer dataset as an example. For details of the dataset, please refer to https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer

We generated pseudo-visium data from the Xenium Prime data measured by 10x Genomics.
Xenium pixel-level counts were aggregated into spots that replicate the Visium array geometry.

.. image:: /_static/Fig_1.png
   :width: 600px
   :align: center
   :alt: Workflow of spEnhance