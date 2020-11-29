Seismophobia Report
================
Trevor Kinsey, Dustin Andrews, Dustin Burnham, Junghoo Kim

2020/11/26

-   [Summary](#summary)
-   [Introduction](#introduction)
-   [Methods](#methods)
-   [Data](#data)
-   [Analysis](#analysis)
-   [Results](#results)
-   [References](#references)

Summary
-------

We built a random forest classification model that predicts whether
someone is afraid of earthquakes based on their demographic information.
The model predicted fear of earthquakes better than a dummy classifier.

Introduction
------------

The damage that earthquakes cause can leave people without food, water,
and shelter. Being prepared for an earthquake before it happens can make
living through the immediate aftermath less traumatic (Paton, Mcclure,
and Buergelt (2006)). Having insurance that covers the damage caused by
earthquakes may reduce the uncertainty and fear that the threat of
earthquakes creates. People who are afraid of earthquakes represent a
group of potential clients for companies selling earthquake preparedness
products and insurance. It has been demonstrated that people who are
more concerned about earthquakes are more likely to have taken
preparatory measures, such as owning a preparedness kit (Dooley et al.
1992).

We aim to predict groups in the population that are afraid of
earthquakes and thus are target demographics for advertising. If a
machine learning algorithm can identify these groups it enables
companies to build a marketing strategy based on this information.

Methods
-------

Data
----

The data for this analysis was downloaded from
<a href="https://github.com/fivethirtyeight/data/tree/master/san-andreas" class="uri">https://github.com/fivethirtyeight/data/tree/master/san-andreas</a>.
It is based on a survey conducted by
[fivethirtyeight.com](https://fivethirtyeight.com/) in May 2015, of 1014
respondents in the US.

The data set contains

-   demographic information (age, gender, household income, region of
    residence in the United States),

-   responses to questions relating to knowledge of and experience with
    earthquakes,

-   self-reported level of fear of earthquakes.

Some preliminary examination of the data shows some relation between the
demographic features and earthquake fear.

<div class="figure" style="text-align: center">

<img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/feature_distributions.png" alt="Fig 1: Distribution of demographic information among survey respondents" width="75%" />
<p class="caption">
Fig 1: Distribution of demographic information among survey respondents
</p>

</div>

The survey respondents were fairly evenly split across ages categories
and genders. The household income distribution was not uniform, but is
likely a reflection of the US population’s income distribution, that had
a median value of $55,775 in 2015 (Bureau 2018). The distribution across
US regions was not uniform, possibly due to the fact that the list of
regions is divided into geographic regions that don’t necessarily have
the same populations. For example, the Pacific region (Washington,
Oregon, and California) has a population of 50.1 million compared to New
England (Connecticut, Massachussets, Maine, New Hampshire, Rhode Island,
and Vermont) has a combined population of 14.7 million (Bureau 2019).

<div class="figure" style="text-align: center">

<img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/feature_distributions_across_response.png" alt="Fig 2: The level of earthquake fear across demograhic features" width="75%" />
<p class="caption">
Fig 2: The level of earthquake fear across demograhic features
</p>

</div>

Younger people tend to report being be more afraid than older people,
while women report being afraid more often than men. The geographic
region with the highest level of fear is the Pacific. Overall there were
more people who were not afraid of earthquakes than were afraid.

<div class="figure" style="text-align: center">

<img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/target_distribution.png" alt="Fig 3: The distribution of earthquake fear among respondents" width="25%" />
<p class="caption">
Fig 3: The distribution of earthquake fear among respondents
</p>

</div>

Analysis
--------

We used a random forest classifier and for the prediction, which in
addition to predicting gives a measure of importance for each feature.
We chose to use only the demographic variables as model features because
they are available in census data without having to conduct another
survey. The prediction target is the self-reported fear of earthquakes,
which we converted from an ordinal variable to a binary variable called
`target`. The class `target` = 0 includes the levels *“not at all
worried”* and *“not so worried”*, while `target` = 1 includes *“somewhat
worried”*, *“very worried”*, and *“extremely worried”*.

Results
-------

Our model was a random forest classifier which we compared to a dummy
classifier that assigned a class randomly based on the target
distribution. Our model had correctly predicted more negative outcomes,
with fewer false positives than the dummy classifier. However it
correctly predicted fewer positive outcomes (people afraid of
earthquakes) than the dummy classifier and had more false negatives.

<div class="figure" style="text-align: center">

<img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/confusion_matrix_DummyClassifier.png" alt="Fig 4: Confusion matrix for dummy classifier and random forest classifier" width="40%" height="40%" /><img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/confusion_matrix_RandomForestClassifier.png" alt="Fig 4: Confusion matrix for dummy classifier and random forest classifier" width="40%" height="40%" />
<p class="caption">
Fig 4: Confusion matrix for dummy classifier and random forest
classifier
</p>

</div>

The combined effects of higher precision and lower recall resulted in
almost identical F1 scores for our model and the dummy classifier.

<div class="figure" style="text-align: center">

<img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/classifier_results_table.png" alt="Fig 5: Classifier F1 scores" width="30%" height="30%" />
<p class="caption">
Fig 5: Classifier F1 scores
</p>

</div>

Of the model’s positive predictions, a greater proportion were correct
than the dummy classifier’s, as characterized by our model’s higher *ROC
AUC*. The model predicted better than if it was guessing at random, but
not a by lot.

<div class="figure" style="text-align: center">

<img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/roc_auc_curve_DummyClassifier.png" alt="Fig 6: ROC curves for dummy classifier and random forest classifier" width="50%" height="50%" /><img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/roc_auc_curve_RandomForestClassifier.png" alt="Fig 6: ROC curves for dummy classifier and random forest classifier" width="50%" height="50%" />
<p class="caption">
Fig 6: ROC curves for dummy classifier and random forest classifier
</p>

</div>

The features that were most important in the prediction task were
household income, living in the Pacific region, and age. (How much
interpretation can we do based on these feature importances? This will
be discussed more next week)

<div class="figure" style="text-align: center">

<img src="C:/Users/Dustin/UBC Masters Data Science/DSCI 522 Data Workflows/seismophobia/visuals/feature_importance.png" alt="Fig 7: Feature importance in random forest classifier" width="80%" height="80%" />
<p class="caption">
Fig 7: Feature importance in random forest classifier
</p>

</div>

Our model does not seem to be very effective in identifying people who
are afraid of earthquakes. A different type of model, such as logistic
regression may yield better results, as well as dealing with the class
imbalance. The greatest potential for improvement lies in obtaining a
more comprehensive data set that contains more demographic features and
a larger sample size.

The R (R Core Team 2019) and Python (Van Rossum and Drake Jr 1995)
programming languages and the following Python packages were used for
this project: pandas (team 2020), sklearn (Pedregosa et al. 2011),
tidyverse (Wickham 2017), knitr (Xie 2014) and docopt
\[<a href="mailto:XXXXXXXXXX@docopt" class="email">XXXXXXXXXX@docopt</a>\].
The code used to perform the analysis and create this report can be
found at:
<a href="https://github.com/UBC-MDS/seismophobia" class="uri">https://github.com/UBC-MDS/seismophobia</a>

References
----------

<div id="refs" class="references hanging-indent">

<div id="ref-bureau_2018">

Bureau, US Census. 2018. “Income and Poverty in the United States:
2015.” *The United States Census Bureau*.
<https://www.census.gov/library/publications/2016/demo/p60-256.html>.

</div>

<div id="ref-bureau_2019">

———. 2019. “State Population Totals: 2010-2019.” *The United States
Census Bureau*.
<https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html>.

</div>

<div id="ref-doi.org/10.1111/j.1559-1816.1992.tb00984.x">

Dooley, David, Ralph Catalano, Shiraz Mishra, and Seth Serxner. 1992.
“Earthquake Preparedness: Predictors in a Community Survey1.” *Journal
of Applied Social Psychology* 22 (6): 451–70.
<https://doi.org/https://doi.org/10.1111/j.1559-1816.1992.tb00984.x>.

</div>

<div id="ref-72124acf2fa84e8aad36e68d0dc4c5e6">

Paton, D, J Mcclure, and Petra Buergelt. 2006. “Natural Hazard
Resilience: The Role of Individual and Household Preparedness.” In
*Disaster Resilience an Integrated Approach*, edited by Douglas Paton
and David Johnston, 105–27. Charles C Thomas Publisher, Ltd.

</div>

<div id="ref-scikit-learn">

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
Grisel, M. Blondel, et al. 2011. “Scikit-Learn: Machine Learning in
Python.” *Journal of Machine Learning Research* 12: 2825–30.

</div>

<div id="ref-R">

R Core Team. 2019. *R: A Language and Environment for Statistical
Computing*. Vienna, Austria: R Foundation for Statistical Computing.
<https://www.R-project.org/>.

</div>

<div id="ref-reback2020pandas">

team, The pandas development. 2020. *Pandas-Dev/Pandas: Pandas* (version
latest). Zenodo. <https://doi.org/10.5281/zenodo.3509134>.

</div>

<div id="ref-van1995python">

Van Rossum, Guido, and Fred L Drake Jr. 1995. *Python Tutorial*. Centrum
voor Wiskunde en Informatica Amsterdam, The Netherlands.

</div>

<div id="ref-tidyverse">

Wickham, Hadley. 2017. *Tidyverse: Easily Install and Load the
’Tidyverse’*. <https://CRAN.R-project.org/package=tidyverse>.

</div>

<div id="ref-knitr">

Xie, Yihui. 2014. “Knitr: A Comprehensive Tool for Reproducible Research
in R.” In *Implementing Reproducible Computational Research*, edited by
Victoria Stodden, Friedrich Leisch, and Roger D. Peng. Chapman;
Hall/CRC. <http://www.crcpress.com/product/isbn/9781466561595>.

</div>

</div>
