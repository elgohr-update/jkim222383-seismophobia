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

## Environment Setup

Configure a conda environment for this repo. From the root of the repo run:

    $ conda env create -f seismophobia_conda_env.yml


For the R scripts, we have exported all session info to `R_session_info.txt` in the root of the folder. This can be used to check if any R package versions fail.

## Script for analysis

The following shell script will run all the scripts required to reproduce our analysis from top to bottom, without need to specify any additional argument. 
From the root of the repo, run:

```bash
bash run_analysis.sh
```
