
# Propulsive-Trailer
This repository is a derivative of the work published in web format at https://trailer.jakobmadgar.com. 

## Intent
The intent of this repoisitory is to provide the code used to generate the simulations and results. The hope is that this can allow others interested in the vehicle dynamics and/or control an example to build off of.

## Usage
After git cloning the repo install the requirements.txt file. This was developed in python 3.9. 

- trailerfunctions.py contains all of the functions used to run the simulations. 
- resultsupdater.py shows examples on how to call the simulations. As noted in the development section below I will make a plotter to view the results like the website.
- The model is stored in `model_driver_eval` method of trailerfunctions.py. This model is built almost entierly from Python [Gekko](https://gekko.readthedocs.io/en/latest/)

## Development/Migration
This code base will be gradually modified to be more user friendly and flexible to use. I will be migrating code from the private repo to this one. The current plan is as follows:
- [ ] Clean up and document main functions in trailerfunctions.py
- [ ] Add controller_comparison.py to generate the different controller profiles in a cleaner fashion.
- [ ] Add a data downloader for the denver to vegas route such that a user can run that simulation.
- [ ] Add a simple streamlit app to allow for data-visualization and knob turning.
