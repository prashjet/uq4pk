This directory contains the code for making all figures in "Uncertainty-aware blob detection with an Application to Integrated-Light Stellar Population Recoveries"
The programs assume that they are executed with "make_plots" as working directory and "PYTHONPATH=/path/to/uq4pk".

This directory is structured as follows:

- run_computations.py: This program runs all the necessary computations from which 
  the plots can be created. It takes a lot of time to finish. 
  The results are stored in the 'out' directory.
- make_plots.py: This program creates the actual plots. It is necessary that 'run_computations.py'
  has been run beforehand. The created plots are stored in the 'plots' directory.
- run_computations_test.py: Runs a test version of all computations. Should be executed 
  before 'run_computations.py' in order to catch errors early. The results are stored in the 'out_test'
  directory.
- make_plots_test.py: Creates demo versions of the plots using the results of 'run_computations_test.py'. 
- src: This directory contains the programs behind the various figures in the paper.
    - blob_detection: Source code for figures 2, 5, 6 and 10.
    - comparison: Source code for figures 15 and 16.
    - computational_aspects: Source code for figures 8 and 9.
    - demo_marginalization: Deprecated.
    - fci: Source code for figures 3 and 4.
    - m54: Source code for figures 11, 12, 13 and 14. This requires that downloading EMILES_BASTI_BASE_BI_FITS from "http://research.iac.es/proyecto/miles/pages/webtools/tune-ssp-models.php" and putting it into "uq4pk/data" (a detailed guide how to do this will be added in the future).
    - mock: Source code for generating the mock datasets.
    - mock_data: Contains various mock datasets.
    - util: Contains auxiliary functions.

For example, to reproduce the plots from the precomputed data (assuming bash):
```bash
cd /path/to/uq4pk/make_plots
PYTHONPATH=/path/to/uq4pk python make_plots.py"
```
