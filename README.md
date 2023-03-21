# Star Spin Up

## Requirements 

```
numpy
matplotlib
scipy
tqdm
```

## Running the Simulation

Simulation of stars spinning up over the course of 15 Myr.

To run, you need to calculate the Kroupa 2013 normalization constant.

```
python3 calc_kroupa2013_imf_norm.py
```

This should make the file `kroupa2013_norm.pickle`.

Then you can run the actual simulation:

```
python3 star_spin_up.py
```

Make sure to change `n_proc` in the file to the number of processes you want to use. The simulation will produce `stars.dat`, which can be analyzed with the jupyter notebook in the `notebooks` folder.

Plots produced by the analysis should be in the notebooks folder.
