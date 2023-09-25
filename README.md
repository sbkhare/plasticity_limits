## Developmental limits and the evolution of plasticity

# plasticity_model.py
* Contains a class that defines an individual and its attributes
* Contains a class that defines a population of individuals
* All used functions have docstrings with descriptions of functions and arguments

# plasticity_function.py
* Reproduces figure 2,
* Different reaction norms for different allelic values

# analyze.py
* Reproduces figures 3, 4, 5, S1, S2, S3, and S4

# run_HPG.py
* Script runs 1000 replicates for a given parameter set and saves the results
* Ran on HiPerGator supercomputer at the University of Florida

# run.py
* Runs a single replicate for a given parameter set
* Used by author to explore behavior of the model

# test.py
* DO NOT USE
* Used by author to explore behavior of the model

# Directory results
* Contains results of 1,000 replicate runs for different parameter sets
* Subdirectory names correspond to parameters: $D_$b_$s
* Subdirectories ending in *_cost are cost cases
* Environmental noise cases are found inside the $D_06_00 folders
* Subdirectories $D_06_00_$tau are not to be used for results
