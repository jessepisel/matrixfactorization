# A Recommender System for Predicting True Vertical Depth to Subsurface Formation Tops
![Prediction Error Maps](https://github.com/jessepisel/matrixfactorization/blob/master/error_maps.png)

This repository contains jupyter notebooks and data to create a recommender system that predicts the true vertical depth (TVD) to subsurface formation tops. Specifically, it predicts the TVD 
and analyzes the predictions for the Teapot Dome oil field in Wyoming USA, and Athabasca Oil Sands in Alberta, CA.

The directories contain the following:
* Munging - Notebook and original Teapot Dome data that are preprocessed in `00_teapot_cleaning.ipynb`
* base_data - Shapefiles for reproducing the figures in the article
* 01_teapot_rec_sys - Notebook and munged Teapot Dome data with associated output
* 02_mannvilee_rec_sys - Notebook and data sources for the Mannville group data
* 03_figures - Notebook to create the figures in the article

## Data Sources
[Teapot Dome from RMOTC](http://s3.amazonaws.com/open.source.geoscience/open_data/teapot/rmotc.tar)

[Mannville Group Well Logs](https://github.com/JustinGOSSES/predictatops/blob/master/demo/mannville_demo_data.zip)

[Mannville Group Picks](https://ags.aer.ca/publications/SPE_006.html)

To run this repository, you need to either clone or download it to your local machine. You will also need to download the Mannville well logs if you want to use them to reproduce the data tables included in the repository.
After cloning, you need to change directories to this directory. To ease in reproducibility we have included a Python virtual environment. Install the virtual environment from a terminal window or anaconda prompt with:

$ conda env create -f recsys.yml

After the virtual environment is installed, activate it and start a jupyter notebook or jupyterlab session

$ conda activate recsys

$ jupyter lab

or

$ jupyter notebook

When the jupyter server has loaded, you can walk through the notebooks in order to reproduce the results from this study.
