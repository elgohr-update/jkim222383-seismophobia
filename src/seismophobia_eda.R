# Autors: Group 11 (Dustin Burnham, Dustin Andrews, Trevor Kinsey, Junghoo Kim)
# Date: November 28th, 2020

"Creates EDA plots for the pre-processed training subset of the seismophobia data set (https://github.com/fivethirtyeight/data/tree/master/san-andreas).
Saves the plots as png files.
Usage: src/seismophobia_eda.R --data_path=<data_path> --out_dir=<out_dir>
Options:
  --data_path=<data_path>     Path (including filename) to training data (csv file)
  --out_dir=<out_dir> Path to directory where the plots should be saved
" -> doc

suppressMessages(library(tidyverse))
suppressMessages(library(docopt))
suppressMessages(library(here))
suppressMessages(library(ggthemes))
suppressMessages(library(testthat))


opt <- docopt(doc)

#' Creates EDA plots and saves as png files in specified directory
#'
#' @param in_dir str path to processed data
#' @param out_dir str path to output visuals directory
#'
#' @return
#'
#' @examples main(in_dir=data/processed/train.csv --out_dir=visuals)
main <- function(in_dir, out_dir) {
  
  # Create out_dir if out_dir and any parent directories in the path do not exist
  dir.create(out_dir, recursive = TRUE)
  
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
    group_by(labeled_target) %>% 
    summarise(target_count = paste0("Count: ", n()),
              target_count_normalized = n() / nrow(earthquake)) %>% 
    ggplot() +
    aes(y = labeled_target,
        x = target_count_normalized,
        label = target_count) +
    geom_bar(stat = "identity", 
             color = "blue", 
             fill = "blue",
             width = 0.4,
             alpha = 0.6) +
    scale_fill_tableau() +
    scale_colour_tableau() +
    geom_text(nudge_x = 0.05) +
    labs(x = "Proportion",
         y = "Survey Response",
         title = "Earthquake Worry Response Distribution",
         subtitle="Training Set") +
    theme(legend.title = element_blank(),
          text = element_text(size=20))

  ggsave(paste0(out_dir, "/target_distribution.png"), 
         width = 12, 
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
    theme(#strip.text = element_text(size=10),
          axis.text = element_text(size = 12),
          axis.text.x = element_text(angle = 90),
          text = element_text(size=25)) +
    coord_flip()
  
  ggsave(paste0(out_dir, "/feature_distributions.png"), 
         width = 10, 
         height = 10)
  
  # Distribution of variables in relation to target class
  num_worried <- sum(earthquake$labeled_target == "worried")
  num_not_worried <- sum(earthquake$labeled_target == "not worried")
  
  earthquake %>% 
    select(-worry_earthquake, -target) %>% 
    pivot_longer(!labeled_target, names_to = "feature", values_to = "value") %>% 
    group_by(labeled_target, feature, value) %>% 
    summarise(group_count = n()) %>% 
    mutate(group_count_normalized = ifelse(labeled_target == "worried", 
                                           group_count / num_worried,
                                           group_count / num_not_worried)) %>% 
    ggplot() +
    aes(x = value, 
        y = labeled_target, 
        fill = group_count) +
    geom_tile(na.rm = TRUE) +
    labs(x = "Features",
         y = "How worried are you about an earthquake?",
         title = "Feature Distributions Across Earthquake Fear",
         fill = "Count") +
    facet_wrap(. ~ feature, scale = "free", ncol = 2) +
    scale_fill_distiller(palette = "Purples", direction = 1) +
    theme_bw() +
    theme(strip.text = element_text(size=10),
          axis.text = element_text(size = 12),
          axis.text.x = element_text(angle = 90),
          text = element_text(size=20)) +
    coord_flip()
  
  ggsave(paste0(out_dir, "/feature_distributions_across_response.png"), 
         width = 12, 
         height = 10)
}


# Paths for running unit tests

FILE_SEP <- .Platform$file.sep

UNIT_TEST_PATH <- here('visuals', 'unit_test')
EDA_PNG1_PATH <- file.path(UNIT_TEST_PATH,
                           "target_distribution.png",
                           fsep = FILE_SEP)
EDA_PNG2_PATH <- file.path(UNIT_TEST_PATH,
                           'feature_distributions.png',
                           fsep = FILE_SEP)
EDA_PNG3_PATH <- file.path(UNIT_TEST_PATH,
                           'feature_distributions_across_response.png',
                           fsep = FILE_SEP)

if (opt[['out_dir']] == UNIT_TEST_PATH) {
  stop('out_dir coincides with unit test path! Cannot run unit tests')
}

# Unit tests
test_that("Unit tests to make sure png files are created", {
  main(opt[['--data_path']], UNIT_TEST_PATH)
  expect_that(UNIT_TEST_PATH, dir.exists, label = 'out_dir is not created')
  expect_that(EDA_PNG1_PATH, file.exists,
              label = paste0(EDA_PNG1_PATH, ' is not created'))
  expect_that(EDA_PNG2_PATH, file.exists,
              label = paste0(EDA_PNG2_PATH, ' is not created'))
  expect_that(EDA_PNG3_PATH, file.exists,
              label = paste0(EDA_PNG3_PATH, ' is not created'))
  # Delete png files created during unit test
  unlink(c(EDA_PNG1_PATH, EDA_PNG2_PATH, EDA_PNG3_PATH))
  # Delete folder created during unit test after checking it's empty
  if (length(list.files(UNIT_TEST_PATH)) == 0) {
    unlink(UNIT_TEST_PATH, recursive = TRUE)
  }
})

main(opt[["--data_path"]], opt[["--out_dir"]])
