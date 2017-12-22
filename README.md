## Introduction
This script is for making [HiPS](https://arxiv.org/abs/1505.02291) data.

## Dependent Packages
* [astropy](http://www.astropy.org)
* [healpy](https://healpy.readthedocs.io/en/latest/)
* I recommend using [Anaconda](https://anaconda.org)
  * astropy is included in anaconda
  * healpy can be installed with "conda"
    ```sh
    conda install -c conda-forge healpy 
    ```


## Usage
```sh
python -m fits2hips.fits2hips -o m31-hips m31-*.fits
```

# See also
* http://aladin.unistra.fr/hips/
* https://arxiv.org/abs/1505.02291
