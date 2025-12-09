.. role:: python(code)
  :language: python
  :class: highlight

Spatial Co-expression Modules Identification
=============================================

To identify spatially organized gene expression patterns, we employed the ``NNMF`` `package <https://github.com/ragnhildlaursen/NNMF>`_ to perform non-negative factorization on the spatial
gene expression matrix, then grouped gene into modules using hierarchical clustering.

.. code-block:: shell

  Rscript run_nnmf.R --path ${prefix} --noSignatures 15 --k 6 --ntop 50

+ **Input**: 
   + ``cnts_train_seed_1.csv``: count matrix used for ``NNMF``.
   + ``locs.csv``: spot location matrix paired with ``cnts_train_seed_1.csv``.
   + ``gene-names.txt``: name of genes selected for clustering and enhancement.
+ **Parameters**:
   + ``${prefix}``: directory to the folder containing the files, i.e. ``data/``.
   + ``--noSignatures``: number of signatures selected after factorization. 
   + ``--k``: number of modules obtained after hierarchical clustering.
+ **Output**: ``gene-names-group.txt``: spatial co-expression modules identified.