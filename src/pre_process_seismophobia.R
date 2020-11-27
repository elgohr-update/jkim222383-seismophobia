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
library(testthat)

# Default value for train_ratio
DEFAULT_TRAIN_RATIO <- 0.7
# Platform-depedent file separator (usually '/' or '\')
FILE_SEP <- .Platform$file.sep

opt <- docopt(doc)

main <- function(input_path, train_ratio, out_dir){
  set.seed(2020)
  
  if (is.null(train_ratio)) {
    train_ratio = DEFAULT_TRAIN_RATIO
  }
  
  if (train_ratio < 0 | train_ratio > 1) stop ('train_ratio should be between 0 and 1!')
  
  train_ratio = as.numeric(train_ratio)
  
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
  
  write_csv(train_data, file.path(out_dir, 'train.csv', fsep = FILE_SEP))
  write_csv(test_data, file.path(out_dir, 'test.csv', fsep = FILE_SEP))

}  

UNIT_TEST_PATH <- here('data', 'unit_test')
TRAIN_DATA_PATH <- file.path(UNIT_TEST_PATH, 'train.csv', fsep = FILE_SEP)
TEST_DATA_PATH <- file.path(UNIT_TEST_PATH, 'test.csv', fsep = FILE_SEP)

if (opt[['out_dir']] == UNIT_TEST_PATH) {
  stop('out_dir coincides with unit test path! Cannot run unit tests')
}

test_that("Unit test for pre_process_seismophobia.R using default train_ratio argument", {
  main(opt[['input_path']], NULL, UNIT_TEST_PATH)
  expect_that(UNIT_TEST_PATH, dir.exists, label = 'out_dir is not created')
  expect_that(TRAIN_DATA_PATH, file.exists, label = 'train.csv is not created')
  expect_that(TEST_DATA_PATH, file.exists, label = 'test.csv is not created')
  df_raw <- read_csv(opt[['input_path']])
  df_train <- read_csv(TRAIN_DATA_PATH)
  df_test <- read_csv(TEST_DATA_PATH)
  expect_equal(nrow(df_train), as.integer(nrow(df_raw) * DEFAULT_TRAIN_RATIO), 
              label = 'train split has wrong number of rows')
  expect_equal(nrow(df_test), nrow(df_raw) - nrow(df_train),
              label = 'test split has wrong number of rows')
  unlink(c(TRAIN_DATA_PATH, TEST_DATA_PATH))
  if (length(list.files(UNIT_TEST_PATH)) == 0) {
    unlink(UNIT_TEST_PATH, recursive = TRUE)
  }
})

test_that("Unit test for pre_process_seismophobia.R using train_ratio argument of 0.3", {
  main(opt[['input_path']], 0.3, UNIT_TEST_PATH)
  expect_that(UNIT_TEST_PATH, dir.exists, label = 'out_dir is not created')
  expect_that(TRAIN_DATA_PATH, file.exists, label = 'train.csv is not created')
  expect_that(TEST_DATA_PATH, file.exists, label = 'test.csv is not created')
  df_raw <- read_csv(opt[['input_path']])
  df_train <- read_csv(TRAIN_DATA_PATH)
  df_test <- read_csv(TEST_DATA_PATH)
  expect_equal(nrow(df_train), as.integer(nrow(df_raw) * 0.3), 
              label = 'train split has wrong number of rows')
  expect_equal(nrow(df_test), nrow(df_raw) - nrow(df_train),
              label = 'test split has wrong number of rows')
  unlink(c(TRAIN_DATA_PATH, TEST_DATA_PATH))
  if (length(list.files(UNIT_TEST_PATH)) == 0) {
    unlink(UNIT_TEST_PATH, recursive = TRUE)
  }
})

test_that("Unit test for pre_process_seismophobia.R using train_ratio argument of 1", {
  main(opt[['input_path']], 1, UNIT_TEST_PATH)
  expect_that(UNIT_TEST_PATH, dir.exists, label = 'out_dir is not created')
  expect_that(TRAIN_DATA_PATH, file.exists, label = 'train.csv is not created')
  expect_that(TEST_DATA_PATH, file.exists, label = 'test.csv is not created')
  df_raw <- read_csv(opt[['input_path']])
  df_train <- read_csv(TRAIN_DATA_PATH)
  df_test <- read_csv(TEST_DATA_PATH)
  expect_equal(nrow(df_train), nrow(df_raw), 
              label = 'train split has wrong number of rows')
  expect_equal(nrow(df_test), 0,
              label = 'test split has wrong number of rows')
  unlink(c(TRAIN_DATA_PATH, TEST_DATA_PATH))
  if (length(list.files(UNIT_TEST_PATH))== 0) {
    unlink(UNIT_TEST_PATH, recursive = TRUE)
  }
})

test_that("train_ratio argument should be between 0 and 1", {
  expect_error(main(opt[['input_path']], -0.1, UNIT_TEST_PATH))
  expect_error(main(opt[['input_path']], 1.1, UNIT_TEST_PATH))
})

main(opt[['input_path']], opt[['train_ratio']], opt[['out_dir']])