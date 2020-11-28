# Autors: Group 11 (Dustin Burnham, Dustin Andrews, Trevor Kinsey, Junghoo Kim)
# Date: November 28th, 2020

"Creates EDA plots for the pre-processes training subset of the seismophobia data set (https://github.com/fivethirtyeight/data/tree/master/san-andreas).
Saves the plots as a pdf and png file.
Usage: src/seismophobia_eda.R --data_path=<data_path> --out_dir=<out_dir>
Options:
  --data_path=<data_path>     Path (including filename) to training data (csv file)
  --out_dir=<out_dir> Path to directory where the plots should be saved
" -> doc

library(tidyverse)
library(docopt)
library(here)
library(ggthemes)
library(testthat)

opt <- docopt(doc)

main <- function(in_dir, out_dir) {
  # Read in data
  earthquake <- read.csv(in_dir)
  
  # Remove survey questions, change age 60 to 60+
  earthquake <- earthquake %>% 
    select(-think_lifetime_big_one, -experience_earthquake, 
           -prepared_earthquake, -familliar_san_andreas,
           -familiar_yellowstone) %>% 
    mutate(labeled_target = ifelse(target == 1, "worried", "not worried")) %>% 
    mutate(age = ifelse(age == "60", "60+", age)) %>% 
    mutate_if(is.character,as.factor)
  
  # Change order of income variables to be numeric
  income_levels <- c("$0 to $9,999", "$10,000 to $24,999", "$25,000 to $49,999",
                     "$50,000 to $74,999", "$75,000 to $99,999", "$100,000 to $124,999",
                     "$125,000 to $149,999", "$150,000 to $174,999", "$175,000 to $199,999",
                     "$200,000 and up", "Prefer not to answer")
  
  earthquake['household_income'] <- factor(earthquake$household_income,
                                           levels = income_levels)

  # Class Distribution
  earthquake %>% 
    ggplot() +
    aes(x = labeled_target) +
    geom_bar(stat = "count", 
             color = "blue", 
             fill = "blue",
             width = 0.4,
             alpha = 0.6) +
    scale_fill_tableau() +
    scale_colour_tableau() +
    labs(x = "Survey Response",
         y = "Count",
         title = "Earthquake Worry Response Distribution",
         subtitle="Training Set") +
    theme(legend.title = element_blank(),
          axis.text.x = element_text(angle = 90)) +
    coord_flip()

  ggsave(paste0(out_dir, "/target_distribution.png"), 
         width = 8, 
         height = 10)
  
  # Generate histograms of each feature
  earthquake %>% 
    select(-target, -worry_earthquake) %>% 
    pivot_longer(!labeled_target, names_to = "feature", values_to = "value") %>% 
    ggplot() +
    aes(x = value) +
    geom_histogram(bins = 10, stat = "count") +
    facet_wrap(. ~ feature, scales = 'free') +
    labs(x = "Features",
         y = "Counts",
         title = "Distributions of Features") +
    theme_bw() +
    theme(strip.text = element_text(size=10),
          axis.text = element_text(size = 8),
          axis.text.x = element_text(angle = 90)) +
    coord_flip()
  
  ggsave(paste0(out_dir, "/feature_distributions.png"), 
         width = 8, 
         height = 10)
  
  # Distribution of variables in relation to target class
  earthquake %>% 
    select(-worry_earthquake, -target) %>% 
    pivot_longer(!labeled_target, names_to = "feature", values_to = "value") %>% 
    add_count(labeled_target, feature, value) %>% 
    ggplot() +
    aes(x = value, 
        y = labeled_target, 
        fill = n) +
    geom_tile(na.rm = TRUE) +
    labs(x = "Features",
         y = "How worried are you about an earthquake?",
         title = "Feature Distributions Across Earthquake Fear") +
    facet_wrap(. ~ feature, scale = "free", ncol = 2) +
    scale_fill_viridis_c(direction=-1) +
    theme_bw() +
    theme(strip.text = element_text(size=10),
          axis.text = element_text(size = 8),
          axis.text.x = element_text(angle = 90)) +
    coord_flip()
  
  ggsave(paste0(out_dir, "/feature_distributions_accross_response.png"), 
         width = 10, 
         height = 10)
  
  # Add tests checking that images were created
  test_that("Test for making sure the plots were properly saved.", {
    expect_that(paste0(out_dir, "/target_distribution.png"), file.exists, label = 'Plot 1 not created!')
    expect_that(paste0(out_dir, "/feature_distributions.png"), file.exists, label = 'Plot 2 not created!')
    expect_that(paste0(out_dir, "/feature_distributions_accross_response.png"), file.exists, label = 'Plot 3 not created!')
  })
}


main(opt[["--data_path"]], opt[["--out_dir"]])
