# Script to execute each step of analysis in proper order with hard coded inputs.
# Run from root of project directory

# Get data from fivethirtyeight url
python src/download_csv.py 'https://raw.githubusercontent.com/fivethirtyeight/data/master/san-andreas/earthquake_data.csv' data/raw/earthquake.csv

# Rename columns, add target column
Rscript src/pre_process_seismophobia.R --input_path=data/raw/earthquake_data.csv --out_dir=data/processed

# Build eda visuals 
Rscript src/seismophobia_eda.R --data_path=data/processed/train.csv --out_dir=visuals