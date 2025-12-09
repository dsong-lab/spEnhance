.. role:: python(code)
  :language: python
  :class: highlight

Validation Set Construction
=============================

To address the common lack of replicates in spatial transcriptomics, where the entire spot-level dataset is often used for training without a hold-out set,
we adopt a ``count splitting`` approach: 
Given a spot-level count matrix :math:`\mathbf{Y}=[Y_{s,g}]\in\mathbb{N}_0^{S\times G}`, we partition each entry as
:math:`Y_{s,g}^{(\mathrm{train})}\sim\mathrm{Binomial}\big(2Y_{s,g}, \tfrac{1}{2}\big)`, :math:`Y_{s,g}^{(\mathrm{val})}=2Y_{s,g}-Y_{s,g}^{(\mathrm{train})}`.

Under a Poisson model assumption, count splitting ensures that :math:`Y_{s,g}^{(\mathrm{train})}` and :math:`Y_{s,g}^{(\mathrm{val})}` 
follow the same distribution as :math:`Y_{s,g}`, and are conditionally independent given their mean value.

This construction therefore yields statistically independent training and validation sets suitable for unbiased assessment of predictive performance.

.. code-block:: shell

   Rscript generate_count_split.R ${prefix} $cnts_train_name $cnts_val_name $seed 

+ **Input**: ``cnts.csv``, file containing count matrix of your ST data, with each row representing a spot and each column representing a gene.
+ **Parameters**:
   + ``${prefix}``: directory to the folder containing the file, i.e. ``data/``.
   + ``$cnts_train_name``: name of the training set. Default name: ``'cnts_train'``.
   + ``$cnts_val_name``: name of the validation set. Default name: ``'cnts_val'``.
   + ``$seed``: random seed use. Default: ``1``.
+ **Output**: ``cnts_train_seed_1.csv`` and ``cnts_val_seed_1.csv``: files containing the training and validation set, respectively.