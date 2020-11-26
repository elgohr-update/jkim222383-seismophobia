# author: Junghoo Kim
# date: 2020-11-26

"Cleans, splits and pre-processes (converts character columns into factors) the fear of earthquake data.
Writes the training and test data to separate csv files.

Usage: src/pre_process_seismophobia.R --input_path=<input_path> [--train_ratio=<train_ratio>] --out_dir=<out_dir>
  
Options:
--input_path=<input_path>       Path (including filename) to raw data
[--train_ratio=<train_ratio>]   Ratio for train split, optional [default: 0.7] 
--out_dir=<out_dir>             Path to directory where the processed data should be written
" -> doc

library(tidyverse)
library(docopt)
library(here)

opt <- docopt(doc)
print(opt)

main <- function(input_path, train_ratio, out_dir){
  DEFAULT_TRAIN_RATIO = 0.7
  set.seed(2020)
  
  if (is.null(train_ratio)) {
    train_ratio = DEFAULT_TRAIN_RATIO
  }
  
  # read data
  earthquake_raw <- read_csv(input_path)
  
  # clean up column names, convert character columns to factors, and reorder the target factor levels
  earthquake_fct <- earthquake_raw %>%
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
  
  # Before we go further, split into train and test splits
  set.seed(42)
  train_index <- sample(1:nrow(earthquake_fct), train_ratio * nrow(earthquake_fct))
  
  train_data <- earthquake_fct %>% filter(row_number() %in% train_index)
  test_data <- earthquake_fct %>% filter(!(row_number() %in% train_index))
  
  try({
    dir.create(out_dir)
  })
  
  write_csv(train_data, paste0(here(out_dir), '/train.csv'))
  write_csv(test_data, paste0(here(out_dir), '/test.csv'))

}  
# TODO: Write tests
  
main(opt[['input_path']], opt[['train_ratio']], opt[['out_dir']])