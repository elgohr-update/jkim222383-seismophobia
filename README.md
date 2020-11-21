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

## Plan

We will evaluate classifier performance using cross validation and pick the classifier that has the lowest cross validation error. Algorithms we will consider include logistic regression, random forest, and naive Bayes. The objective is to build a binary classifier that can predict a person's fear of earthquakes. We will evaluate the final classifier with the F1 score and area under the receiver operating characteristic curve.

## Exploratory Data Analysis Plan

We will evaluate the raw data first for missing values, and determine if an imputation scheme is appropriate. Next we will look at linear correlations of the respective features in a correlation heat map. We will also look at a histogram of the target class distribution to get an approximate idea of the frequency of different responses.

EDA document can be found here: [EDA Notebook](./src/seismophobia.md)

## Communication of Results

We will wrap our analysis in a reproducible code document that can download the data locally, produce the EDA report and then produce a final report of classifier tuning and performance. We will also discuss next steps we might take if we were to ask a slightly different question or use more data. The report will have a summary table of classifier performance, in addition to plots that show classifier performance. We will create a `Makefile` and `Dockerfile` capable of recreating our analysis on an end user's machine.
