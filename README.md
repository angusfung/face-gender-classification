# classification

This is a an adaption of faces.py using the PEP8 coding style. 

## Download

You must download the dataset before running any parts of the code. 

```
./classification.py --download --log INFO 
```

This will download from the FaceScrub dataset into two directories under `dataset/`: `uncropped` (raw images) and `cropped` (grey-scaled images).

## Generate training, test, validation sets

```
./classification.py --part 2
```
This process creates non-overlapping, reproducible (seeded) datasets for the actors/actresses Drescher, Ferrera, Chenoweth, Baldwin, Hader, and Carell.




