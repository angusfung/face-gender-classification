# classification

This is a an adaption of faces.py using the PEP8 coding style.  Rewrite this and describe the project better.

## Download

You must download the dataset before running any parts of the code.

```
./classification.py --download --log INFO 
```

This will download from the FaceScrub dataset into two directories under `dataset/`: `uncropped` (raw images) and `cropped` (grey-scaled images).

### Generate training, test, validation sets

```
./classification.py --part 2
```
This process creates non-overlapping, reproducible (seeded) datasets for the actors/actresses Drescher, Ferrera, Chenoweth, Baldwin, Hader, and Carell.

### Create a classifier between Hader and Carell
```
./classification.py --part 3 --log INFO
```
The implementation details are in the PDF. The optimal parameters for gradient descent were determined by using the `--optimal` option. A sample report of that is under `logs/optimal_params`. Those optimal parameters are then used in gradient descent to determine the optimized theta value, which is subsequently saved into `part3.pkl`.  

The classifier is then used to classfy 10 images of hader and carell, and the results are printed to the terminal.

### Visualizing theta
```
./classification.py --part 4 --log INFO
```
This visualizes the optimized theta by loading the saved theta value from part 3, for both the full training set and a two-image training set.

### Gender classification
```
./classification.py --part 5 --log INFO
```
Trains the classifier on images of `6` actors / actresses and then tests the performance on `10` images of `6` different actors / actresses. The size of the training size is varied from `10` to `70` and the performance is plotted.

### Multi-actor classification using One Hot Encoding
```
./classification.py --part 7 --log INFO
```
Trains the classifier on images of `6` actors / actresses and then attempts to classify them on the validation and test set.





