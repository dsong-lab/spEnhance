# spEnhance
`spEnhance` is a computational framework that reconstructs spatial gene expression with super-resolution from spot-level data while quantifying the confidence in its predictions. spEnhance achieves higher prediction accuracy by integrating both histology and single-cell RNA-seq (scRNA-seq) data, and employs a novel way to construct validation sets for trustworthy prediction.

## Get Started
To run the demo,
```
# Use Python 3.9 or above
./run_demo.sh
```
Using GPUs is recommended.

### Data format
+ `he-raw.jpg`: Raw H&E image.
+ `cnts.csv`: Spatial gene count matrix.
  + Row 1: Gene names.
  + Row 2 and after: Each row represents a spot.
  + Column 1: Spot ID
  + Column 2 and after: Each column represents a gene.
+ `locs-raw.csv`: Spot location.
  + Row 1: Header.
  + Row 2 and after: Each row represents a spot. Number and order of the spots should match `cnts.csv`
  + Column 1: Spot ID.
  + Column 2: x-coordinate of the spot (horizontal axis). Must be in the same space as column of the array indices of pixels in `he-raw.jpg`.
  + Column 3: y-coordinate of the spot (vertical axis). Must be in the same space as row of the array indices of pixels in `he-raw.jpg`.
+ `pixel-size-raw.txt`: Side length (in micrometers) of a pixel in `he-raw.jpg`.
+ `radius-raw.txt`: Number of pixels covered by spot radius in `he-raw.jpg`.

### Documentation
A step-by-step tutorial for major steps is available at https://spenhance-documentation.readthedocs.io/en/latest/imputation.html

### Demo data 
Demo data deposited at https://doi.org/10.6084/m9.figshare.30827999

Please download the files and put them under `data/` directory.
