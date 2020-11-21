Outline
-------

This document will provide a first pass at an EDA of the
`fivethirtyeight` San Andreas Earthquake data set.

    library(tidyverse)
    library(here)

    ## Warning: package 'here' was built under R version 4.0.3

    library(ggthemes)

    ## Warning: package 'ggthemes' was built under R version 4.0.3

    earthquake <- read_csv(here('data', 'raw', 'earthquake_data.csv'))

    # Before we go further, split into 30% test, 70% train
    set.seed(42)
    train_ratio = 0.7
    train_index <- sample(1:nrow(earthquake), train_ratio * nrow(earthquake))

    earthquake_fct <- earthquake %>%
      filter(row_number() %in% train_index) %>%
      rename(
        worry_earthquake = "In general, how worried are you about earthquakes?",
        worry_big_one = "How worried are you about the Big One, a massive, catastrophic earthquake?",
        think_lifetime_big_one = 'Do you think the "Big One" will occur in your lifetime?',
        experience_earthquake = "Have you ever experienced an earthquake?",
        prepared_earthquake = "Have you or anyone in your household taken any precautions for an earthquake (packed an earthquake survival kit, prepared an evacuation plan, etc.)?",
        familliar_san_andreas = "How familiar are you with the San Andreas Fault line?",
        familiar_yellowstone = "How familiar are you with the Yellowstone Supervolcano?",
        age = "Age",
        gender = "What is your gender?",
        household_income = "How much total combined money did all members of your HOUSEHOLD earn last year?",
        us_region = "US Region"
      ) %>%
      mutate_all(na_if, "") %>%
      mutate_if(is.character, as.factor) %>%
      select(-worry_big_one) %>%
      mutate(worry_earthquake = fct_relevel(worry_earthquake, c("Extremely worried", "Very worried", "Somewhat worried", "Not so worried", "Not at all worried")))

    earthquake_fct %>%  
      head(10) %>% 
      knitr::kable(caption = "Head of Earthquake Data Set - Train Split ")

<table style="width:100%;">
<caption>Head of Earthquake Data Set - Train Split</caption>
<colgroup>
<col style="width: 10%" />
<col style="width: 12%" />
<col style="width: 14%" />
<col style="width: 10%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 4%" />
<col style="width: 3%" />
<col style="width: 11%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">worry_earthquake</th>
<th style="text-align: left;">think_lifetime_big_one</th>
<th style="text-align: left;">experience_earthquake</th>
<th style="text-align: left;">prepared_earthquake</th>
<th style="text-align: left;">familliar_san_andreas</th>
<th style="text-align: left;">familiar_yellowstone</th>
<th style="text-align: left;">age</th>
<th style="text-align: left;">gender</th>
<th style="text-align: left;">household_income</th>
<th style="text-align: left;">us_region</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">Not so familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">Prefer not to answer</td>
<td style="text-align: left;">New England</td>
</tr>
<tr class="even">
<td style="text-align: left;">Somewhat worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Not at all familiar</td>
<td style="text-align: left;">Not at all familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">$75,000 to $99,999</td>
<td style="text-align: left;">East North Central</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not so worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Very familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">$10,000 to $24,999</td>
<td style="text-align: left;">Pacific</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not so worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Very familiar</td>
<td style="text-align: left;">Not so familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">$25,000 to $49,999</td>
<td style="text-align: left;">West South Central</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Very worried</td>
<td style="text-align: left;">Yes</td>
<td style="text-align: left;">Yes, one or more major ones</td>
<td style="text-align: left;">Yes</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Not so familiar</td>
<td style="text-align: left;">Not at all familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">Prefer not to answer</td>
<td style="text-align: left;">South Atlantic</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Very familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">Prefer not to answer</td>
<td style="text-align: left;">East North Central</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Not at all familiar</td>
<td style="text-align: left;">Not so familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">$10,000 to $24,999</td>
<td style="text-align: left;">East North Central</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more major ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">Not so familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">$50,000 to $74,999</td>
<td style="text-align: left;">West North Central</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Extremely familiar</td>
<td style="text-align: left;">Extremely familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">$25,000 to $49,999</td>
<td style="text-align: left;">East South Central</td>
</tr>
</tbody>
</table>

    earthquake_fct %>%  
      tail(10) %>% 
      knitr::kable(caption = "Tail of Earthquake Data Set")

<table style="width:100%;">
<caption>Tail of Earthquake Data Set</caption>
<colgroup>
<col style="width: 10%" />
<col style="width: 12%" />
<col style="width: 14%" />
<col style="width: 10%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 4%" />
<col style="width: 3%" />
<col style="width: 11%" />
<col style="width: 10%" />
</colgroup>
<thead>
<tr class="header">
<th style="text-align: left;">worry_earthquake</th>
<th style="text-align: left;">think_lifetime_big_one</th>
<th style="text-align: left;">experience_earthquake</th>
<th style="text-align: left;">prepared_earthquake</th>
<th style="text-align: left;">familliar_san_andreas</th>
<th style="text-align: left;">familiar_yellowstone</th>
<th style="text-align: left;">age</th>
<th style="text-align: left;">gender</th>
<th style="text-align: left;">household_income</th>
<th style="text-align: left;">us_region</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">Not so worried</td>
<td style="text-align: left;">Yes</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Very familiar</td>
<td style="text-align: left;">Not at all familiar</td>
<td style="text-align: left;">60</td>
<td style="text-align: left;">Female</td>
<td style="text-align: left;">$25,000 to $49,999</td>
<td style="text-align: left;">East North Central</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not so worried</td>
<td style="text-align: left;">Yes</td>
<td style="text-align: left;">Yes, one or more major ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Extremely familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">60</td>
<td style="text-align: left;">Female</td>
<td style="text-align: left;">$75,000 to $99,999</td>
<td style="text-align: left;">South Atlantic</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Not at all familiar</td>
<td style="text-align: left;">Not at all familiar</td>
<td style="text-align: left;">18 - 29</td>
<td style="text-align: left;">Female</td>
<td style="text-align: left;">$10,000 to $24,999</td>
<td style="text-align: left;">East North Central</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Very familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">45 - 59</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">$75,000 to $99,999</td>
<td style="text-align: left;">East North Central</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">60</td>
<td style="text-align: left;">Female</td>
<td style="text-align: left;">Prefer not to answer</td>
<td style="text-align: left;">New England</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Very familiar</td>
<td style="text-align: left;">Not so familiar</td>
<td style="text-align: left;">60</td>
<td style="text-align: left;">Male</td>
<td style="text-align: left;">Prefer not to answer</td>
<td style="text-align: left;">Pacific</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not so worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more major ones</td>
<td style="text-align: left;">Yes</td>
<td style="text-align: left;">Extremely familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">60</td>
<td style="text-align: left;">Female</td>
<td style="text-align: left;">$50,000 to $74,999</td>
<td style="text-align: left;">Pacific</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not so worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">30 - 44</td>
<td style="text-align: left;">Female</td>
<td style="text-align: left;">Prefer not to answer</td>
<td style="text-align: left;">Middle Atlantic</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">Somewhat familiar</td>
<td style="text-align: left;">30 - 44</td>
<td style="text-align: left;">Female</td>
<td style="text-align: left;">$50,000 to $74,999</td>
<td style="text-align: left;">East North Central</td>
</tr>
<tr class="even">
<td style="text-align: left;">Somewhat worried</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">Yes, one or more minor ones</td>
<td style="text-align: left;">No</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
<td style="text-align: left;">NA</td>
</tr>
</tbody>
</table>

    earthquake_fct$worry_earthquake %>%
      levels()

    ## [1] "Extremely worried"  "Very worried"       "Somewhat worried"  
    ## [4] "Not so worried"     "Not at all worried"

    earthquake_fct %>% 
      group_by(worry_earthquake) %>% 
      summarize(count = n()) %>% 
      knitr::kable(caption="Count of Different Target Levels in Training Set")

    ## `summarise()` ungrouping output (override with `.groups` argument)

<table>
<caption>Count of Different Target Levels in Training Set</caption>
<thead>
<tr class="header">
<th style="text-align: left;">worry_earthquake</th>
<th style="text-align: right;">count</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">Extremely worried</td>
<td style="text-align: right;">35</td>
</tr>
<tr class="even">
<td style="text-align: left;">Very worried</td>
<td style="text-align: right;">38</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Somewhat worried</td>
<td style="text-align: right;">156</td>
</tr>
<tr class="even">
<td style="text-align: left;">Not so worried</td>
<td style="text-align: right;">221</td>
</tr>
<tr class="odd">
<td style="text-align: left;">Not at all worried</td>
<td style="text-align: right;">259</td>
</tr>
</tbody>
</table>

    earthquake_fct %>%
      is.na() %>%
      colSums() %>%
      knitr::kable(col.names = "Count of NA's in Feature", caption="NA Count by Feature In Training Set")

<table>
<caption>NA Count by Feature In Training Set</caption>
<thead>
<tr class="header">
<th style="text-align: left;"></th>
<th style="text-align: right;">Count of NA’s in Feature</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: left;">worry_earthquake</td>
<td style="text-align: right;">0</td>
</tr>
<tr class="even">
<td style="text-align: left;">think_lifetime_big_one</td>
<td style="text-align: right;">0</td>
</tr>
<tr class="odd">
<td style="text-align: left;">experience_earthquake</td>
<td style="text-align: right;">5</td>
</tr>
<tr class="even">
<td style="text-align: left;">prepared_earthquake</td>
<td style="text-align: right;">5</td>
</tr>
<tr class="odd">
<td style="text-align: left;">familliar_san_andreas</td>
<td style="text-align: right;">10</td>
</tr>
<tr class="even">
<td style="text-align: left;">familiar_yellowstone</td>
<td style="text-align: right;">10</td>
</tr>
<tr class="odd">
<td style="text-align: left;">age</td>
<td style="text-align: right;">10</td>
</tr>
<tr class="even">
<td style="text-align: left;">gender</td>
<td style="text-align: right;">10</td>
</tr>
<tr class="odd">
<td style="text-align: left;">household_income</td>
<td style="text-align: right;">10</td>
</tr>
<tr class="even">
<td style="text-align: left;">us_region</td>
<td style="text-align: right;">32</td>
</tr>
</tbody>
</table>

    earthquake_fct %>% 
      ggplot() +
        aes(x = worry_earthquake ,
            color = worry_earthquake,
            fill = worry_earthquake) +
      geom_histogram(stat = "count") +
      scale_fill_tableau() +
      scale_colour_tableau() +
      labs(x = "Worry Response",
           y = "Count",
           title = "Earthquake Worry Response Distribution",
           subtitle="Training Set") +
      theme(legend.title = element_blank(),
            axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            axis.ticks.x=element_blank())

    ## Warning: Ignoring unknown parameters: binwidth, bins, pad

![](seismophobia_eda_files/figure-markdown_strict/-%20Class%20Distribution-1.png)

    earthquake_fct %>% 
      pivot_longer(!worry_earthquake, names_to = "feature", values_to = "value") %>% 
      add_count(worry_earthquake, feature, value) %>% 
      ggplot() +
        aes(x = value, 
            y = worry_earthquake, 
            fill = n) +
      geom_tile(na.rm = TRUE) +
      xlab("Feature") + 
      ylab("How worried are you about an earthquake?") + 
      facet_wrap(. ~ feature, scale = "free", ncol = 3) +
      scale_fill_viridis_c(direction=-1) +
      theme_bw() +
      theme(strip.text = element_text(size=10),
            axis.text = element_text(size = 8),
            axis.text.x = element_text(angle = 90))

![](seismophobia_eda_files/figure-markdown_strict/-%202D%20Histograms-1.png)
