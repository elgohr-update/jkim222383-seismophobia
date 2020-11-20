library(rvest)
library(here)
library(tidyverse, quietly = TRUE)
library(janitor, quietly=TRUE)

region_url <- "https://www.nationsonline.org/oneworld/US-states-by-area.htm"
earthquake_count_url <- "https://www.usgs.gov/natural-hazards/earthquake-hazards/lists-maps-and-statistics"

xpath_regions <- '//*[@id="statelist"]'
xpath_earthquake_counts <- '//*[@id="block-system-main"]/div/div/div[2]/div/section[1]/div[3]/div/div/div/div/div/div[5]/div/div/div/div/table'


# Table of State -> Region lookups
regions <- region_url %>% 
  xml2::read_html() %>% 
  html_nodes(xpath=xpath_regions) %>% 
  html_table() %>% 
  .[[1]] %>% 
  clean_names() %>% 
  select(census_region, state)

# Table of Earthquakes per State
earthquake_counts <- earthquake_count_url %>% 
  xml2::read_html() %>% 
  html_nodes(xpath=xpath_earthquake_counts) %>% 
  html_table() %>% 
  .[[1]] %>% 
  clean_names()


earthquakes_by_region <- earthquake_counts %>% 
  left_join(regions, by=c("states"="state")) %>% 
  pivot_longer(cols=-c(states,census_region)) %>% 
  rename(us_region = census_region) %>% 
  group_by(us_region) %>% 
  summarize(total_earthquakes_2010_2015 = sum(value), .groups='keep')

earthquake <- read_csv(here('data', 'raw', 'earthquake_data.csv')) %>% 
  clean_names()

earthquake_plus_counts <- earthquake %>% 
  left_join(earthquakes_by_region, by='us_region')

write_csv(earthquake_plus_counts,here('data','processed','earthquakes_plus_counts.csv'))

