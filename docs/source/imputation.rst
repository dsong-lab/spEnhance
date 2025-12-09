.. role:: python(code)
  :language: python
  :class: highlight

Trustworthy prediction of super-resolution spatial gene expression
==================================================================

To predict pixel-level spatial gene expression :math:`\mathbf{Z}\in\mathbb{R}^{H_1\times W_1\times G}`, 
we train a deep neural network that maps the fused histology-gene embeddings 
:math:`\mathbf{F}=\mathrm{Concat}(\mathbf{U},\mathbf{V})\in\mathbb{R}^{H_1\times W_1\times (D+G_1)}`
to :math:`\mathbf{Z}`, using the spot-level count matrix :math:`\mathbf{Y}` for weak supervision.

Our model is a graph convolutional network (GCN) with two shared graph-convolution layers (512 hidden units each).
After the shared layers, each module (as indicated before) is predicted by its own head (a linear layer mapping 512 to the number of genes in the module). 
A dropout layer (p=0.5) is applied between the shared representation and each module-specific head to reduce overfitting. 

To train and validate the model, and used the trained models to conduct prediction, run the following code:

.. code-block:: shell

   python impute_slide_val_MIL.py ${prefix} \
        --cnts_train_name ${cnts_train_name}_seed_${seed}.csv \
        --cnts_val_name ${cnts_val_name}_seed_${seed}.csv \
        --epochs=150 --device='cuda' --n_states=5

+ **Input**:
   + ``embeddings-combined.pickle``: combined gene expression and histology features.
   + ``cnts_train_seed_1.csv``: training set.
   + ``cnts_val_seed_1.csv``: validation set.
   + ``locs.csv``: spot location matrix paired with training and validation set.
   + ``gene-names.txt``: genes selected for enhancement.
   + ``radius.txt``: rescaled spot radius.
+ **Parameters**:
   + ``${prefix}``: directory to the folder containing the image, i.e. ``data/``.
   + ``--cnts_train_name``: file name of the training data
   + ``--cnts_val_name``: file name of the validation data
   + ``--epochs``: number of epochs when training the model
   + ``--device``: device used for training, validation and prediction. The use of GPU is strongly recommended.
   + ``--n_states``: number of states (number of independent models trained, validated and used for prediction)
+ **Output**: ``Prediction_val/cnts-super/``: directory containing enhancement results, with each gene saved as ``GENE.pickle``.

.. image:: /_static/imputation.png
   :width: 600px
   :align: center
   :alt: Imputation