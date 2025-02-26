#!/bin/bash

echo "Running get_data.py..."
python ./code/python/get_data.py

echo "Running compute_derived_variables.py..."
python ./code/python/compute_derived_variables.py

echo "Running get_methods_info.py..."
python ./code/python/get_methods_info.py

echo "Running cogsci_results.Rmd..."
Rscript -e "rmarkdown::render('code/R/cogsci_results.Rmd')"


echo "All tasks completed."
