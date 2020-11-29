---
title: "Project Proposal"
author: "Group 11 - Junghoo Kim, Trevor Kinsey, Dustin Burnham, Dustin Andrews"
date: "11/19/2020"
output: html_document
---

## Dataset

We'll be working with the San Andreas Earthquake data set from [`fivethirtyeight`](https://github.com/fivethirtyeight/data/tree/master/san-andreas) which collected people's relative fear of earthquakes along with other demographic attributes such as age, gender, household income and region.

## Purpose

We aim to determine a model for predicting a person's fear of earthquakes, given demographic features about that person. We will investigate a binary classifier, that can predict if a person has fear about earthquakes (seismophobia) given their prior experience with earthquakes and demographic attributes.

## Report

The final report can be found [here](https://github.com/UBC-MDS/seismophobia/blob/main/doc/seismophobia_report.md). 

## Environment Setup

After cloning this GitHub repository, install the required conda dependencies by configuring a conda environment for this repo. From the root of the repo run:

    $ conda env create -f seismophobia_conda_env.yml


For the R scripts, we have exported all session info to `R_session_info.txt` in the root of the folder. This can be used to check if any R package versions fail.

## Usage

After the required environments are set up, the following shell script will run all the scripts required to reproduce our analysis from top to bottom, without need to specify any additional argument. 
From the root of the repo, run:

```bash
bash run_analysis.sh
```
## License

UBC-MDS/seismophobia is licensed under the MIT License. If re-using/re-mixing, please provide attribution and link to this webpage. 

More information regarding license can be found [here](https://github.com/UBC-MDS/seismophobia/blob/main/LICENSE).

# References

Bureau, US Census. 2018. “Income and Poverty in the United States: 2015.” The United States Census Bureau. https://www.census.gov/library/publications/2016/demo/p60-256.html.

———. 2019. “State Population Totals: 2010-2019.” The United States Census Bureau. https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html.

Dooley, David, Ralph Catalano, Shiraz Mishra, and Seth Serxner. 1992. “Earthquake Preparedness: Predictors in a Community Survey1.” Journal of Applied Social Psychology 22 (6): 451–70. https://doi.org/https://doi.org/10.1111/j.1559-1816.1992.tb00984.x.

Paton, D, J Mcclure, and Petra Buergelt. 2006. “Natural Hazard Resilience: The Role of Individual and Household Preparedness.” In Disaster Resilience an Integrated Approach, edited by Douglas Paton and David Johnston, 105–27. Charles C Thomas Publisher, Ltd.

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in Python.” Journal of Machine Learning Research 12: 2825–30.

R Core Team. 2019. R: A Language and Environment for Statistical Computing. Vienna, Austria: R Foundation for Statistical Computing. https://www.R-project.org/.

team, The pandas development. 2020. Pandas-Dev/Pandas: Pandas (version latest). Zenodo. https://doi.org/10.5281/zenodo.3509134.

Van Rossum, Guido, and Fred L Drake Jr. 1995. Python Tutorial. Centrum voor Wiskunde en Informatica Amsterdam, The Netherlands.

Wickham, Hadley. 2017. Tidyverse: Easily Install and Load the ’Tidyverse’. https://CRAN.R-project.org/package=tidyverse.

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research in R.” In Implementing Reproducible Computational Research, edited by Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman; Hall/CRC. http://www.crcpress.com/product/isbn/9781466561595.
