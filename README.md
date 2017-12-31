# classification

This is a an adaption of faces.py using the PEP8 coding style. 

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
The implementation details are in the PDF. 
The optimal parameters for gradient descent were determined by adding the `--optimal` option. A sample report of that is under `logs/optimal_params`. Those optimal parameters are then used in gradient descent to determine the optimized theta value, which is then saved into `part3.pkl`.




