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
  
  # Generate histograms of each feature
  earthquake %>% 
    pivot_longer(!worry_earthquake, names_to = "feature", values_to = "value") %>% 
    ggplot() +
      aes(x = value) +
    geom_histogram(bins = 10, stat = "count") +
    facet_wrap(. ~ feature, scales = 'free_x') +
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
    xlab("Feature") + 
    ylab("How worried are you about an earthquake?") + 
    facet_wrap(. ~ feature, scale = "free", ncol = 3) +
    scale_fill_viridis_c(direction=-1) +
    theme_bw() +
    theme(strip.text = element_text(size=10),
          axis.text = element_text(size = 8),
          axis.text.x = element_text(angle = 90))
  
  ggsave(paste0(out_dir, "/feature_distributions_accross_response.png"), 
         width = 8, 
         height = 10)
}

main(opt[["--in_dir"]], opt[["--out_dir"]])