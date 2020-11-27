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

opt <- docopt(doc)

main <- function(in_dir, out_dir) {
  # Read in data
  earthquake <- read.csv(in_dir)
  
  # Remove survey questions
  earthquake <- earthquake %>% 
    select(-think_lifetime_big_one, -experience_earthquake, 
           -prepared_earthquake, -familliar_san_andreas,
           -familiar_yellowstone)
  
  # Class Distribution
  earthquake %>% 
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
  
  ggsave(paste0(out_dir, "/target_distribution.png"), 
         width = 8, 
         height = 10)
  
  # Generate histograms of each feature
  earthquake %>% 
    pivot_longer(!worry_earthquake, names_to = "feature", values_to = "value") %>% 
    ggplot() +
      aes(x = value) +
    geom_histogram(bins = 10, stat = "count") +
    facet_wrap(. ~ feature, scales = 'free_x') +
    labs(x = "Features",
         y = "Counts",
         title = "Distributions of Features") +
    theme_bw() +
    theme(strip.text = element_text(size=10),
          axis.text = element_text(size = 8),
          axis.text.x = element_text(angle = 90))
  
  ggsave(paste0(out_dir, "/feature_distributions.png"), 
         width = 8, 
         height = 10)
  
  # Distribution of variables in relation to target class
  earthquake %>% 
    pivot_longer(!worry_earthquake, names_to = "feature", values_to = "value") %>% 
    add_count(worry_earthquake, feature, value) %>% 
    ggplot() +
    aes(x = value, 
        y = worry_earthquake, 
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
          axis.text.x = element_text(angle = 90))
  
  ggsave(paste0(out_dir, "/feature_distributions_accross_response.png"), 
         width = 10, 
         height = 10)
  
  
}

main(opt[["--data_path"]], opt[["--out_dir"]])