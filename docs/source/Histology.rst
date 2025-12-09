.. role:: python(code)
  :language: python
  :class: highlight

Histology Feature Extraction
=============================

We extract histological features using a dual-model strategy. Specifically, we employed `HIPT <https://github.com/mahmoodlab/HIPT>`_, 
a hierarchical vision transformer that captures multi-scale tissue architecture, and `UNI <https://github.com/mahmoodlab/UNI>`_, 
a universal pathology foundation model pretrained across diverse histology cohorts. 
Leveraging two complementary models allows us to integrate both global contextual representations and fine-grained local morphology.

Global histological feature extraction using HIPT
------------------------------------------------------

First, download model weights of HIPT by running ``download_pretrained_vit.sh``.

.. code-block:: shell

   python extract_features_vit.py ${prefix} --device='cuda'

+ **Input**: ``he.jpg``, preprocessed H&E image paired with your ST data.
+ **Parameters**:
   + ``${prefix}``: directory to the folder containing the image, i.e. ``data/``.
   + ``--device``: choosing device to use, either ``cude`` or ``cpu``. 
+ **Output**: ``embeddings-hist-vit.pickle``: pickle file containing global and local image features.

The use of GPU is highly recommended.

Fine-grained histological feature extraction using UNI
------------------------------------------------------

First, request access to the UNI model weights from the Huggingface model page at https://huggingface.co/mahmoodlab/UNI.

.. code-block:: shell

   python extract_features_uni.py ${prefix} --login='LOGIN'

+ **Input**: ``he.jpg``, preprocessed H&E image paired with your ST data.
+ **Parameters**:
   + ``${prefix}``: directory to the folder containing the image, i.e. ``data/``.
   + ``--login``: replace with your own login to access UNI weights. 
+ **Output**: ``embeddings-hist-uni.pickle``: pickle file containing fine-grained image features.

The use of GPU is highly recommended.

Integration of global and fine-grained histological features
------------------------------------------------------------

Combining two complementary embeddings allows us to integrate global contextual representations (HIPT) with fine-grained local morphology (HIPT \& UNI). 
A unified histology embedding is constrcuted by applying principal component analysis (PCA) to each model's features (retaining components that explain 
at least 99% of the variance) and then concatenating the reduced embeddings.

.. code-block:: shell

   python merge_feature.py ${prefix} --method='pca'

+ **Input**: ``embeddings-hist-vit.pickle`` and ``embeddings-hist-uni.pickle``.
+ **Parameters**:
   + ``${prefix}``: directory to the folder containing the files, i.e. ``data/``.
   + ``--method``: method for dimension reduction, options contain ``pca``, ``nmf`` and ``ica``. ``pca`` is recommended.
+ **Output**: ``embeddings-hist-merged.pickle``: pickle file containing merged image features.

.. image:: /_static/histology.png
   :width: 600px
   :align: center
   :alt: Histology features
