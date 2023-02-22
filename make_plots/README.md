This directory contains the code for making all figures in "Uncertainty-aware blob detection with an 
Application to Integrated-Light Stellar Population Recoveries"
The programs assume that they are executed with `make_plots.py` as working directory and `PYTHONPATH=/path/to/uq4pk`.

- `run_computations.py`: This program runs all the necessary computations from which 
  the plots can be created. There are three modes of deployment: `test`, `base` and `final`. 
`final` reproduces the plots in the paper, but takes a lot of time to finish. 
  The results are stored in the `out` directory.
- `make_plots.py`: This program creates the actual plots. It is necessary that `run_computations.py`
  has been run beforehand, with the matching mode. The created plots are stored in the `plots` directory.
- `src`: This directory contains the programs behind the various figures in the paper.
    - `intro`: Source code for figure 1.
    - `blob_detection`: Source code for figures 2,4, 5 and 6.
    - `fci`: Source code for figure 3.
    - `m54`: Source code for figures 8 - 12.
    - `mock`: Source code for generating the mock datasets.
    - `mock_data`: Contains various mock datasets.
    - `svd_mcmc`: Source code figure 7.
    - `util`: Contains auxiliary functions.

For example, to reproduce the plots from the precomputed data (in bash):
```bash
cd /path/to/uq4pk/make_plots
PYTHONPATH=/path/to/uq4pk python make_plots.py"
```


### Note on the M54 data

Reproducing the results for the M54 data requires downloading `EMILES_BASTI_BASE_BI_FITS` from
the MILES website. 
The files are not provided with the repository since they are too large. Here is a short guide on how to download
the correct files:

1. Go to the website [http://research.iac.es/proyecto/miles/pages/webtools/tune-ssp-models.php]().
2. In the 'Input parameters' section, choose 'E-MILES' in the drop down menu under 'SSP Models'.
3. Leave all other settings as default.
4. Click 'Submit Query'.
5. This leads to an FTP server. After connecting to that server using suitable software (e.g. FileZilla), 
download the archive "E-MILES/EMILES_BASTI_BASE_BI_FITS.tar.gz" and extract it in the local `data` directory.
The source code expects the contents of this archive to be in `data/EMILES_BASTI_BASE_BI_FITS`.
