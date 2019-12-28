# Project description
Pytexdiv is a small python code for fast kullback-Leibller divergence computation between two statistical models using PyOpenCL. I used this program for my research work during my PhD thesis to benchmark different statistical models when I had needed to avoid wating hours or days to get a retrieval rate when using a large dataset. Pytexdiv uses OpenCL in the background to perform the parallel divergence computation.

# Usage
The program runs on command line with the following arguments:
* -f: The path to the file that contains all the models (where each line is a model expressed with a set of features)
* -m: The model. Currently only two models are supported: Generalized Gaussian distribution (ggd) and Weibull (weibull)
* -d: The divergence type we would like to compute (kld or csd). Currently only KLD is supported.
* -gpu: Use accelerated computation using GPU programming (0 or 1)
* -mci: Use Monte-Carlo integration for non-analytic divergences (0 or 1)
* -s: Save results in a text file (0 or 1)