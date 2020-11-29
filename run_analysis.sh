# Script to execute each step of analysis in proper order with hard coded inputs.
# Run from root of project directory

# Get data from fivethirtyeight url
python src/download_csv.py 'https://raw.githubusercontent.com/fivethirtyeight/data/master/san-andreas/earthquake_data.csv' data/raw/earthquake.csv

# Rename columns, add target column
Rscript src/pre_process_seismophobia.R --input_path=data/raw/earthquake_data.csv --out_dir=data/processed

# Build eda visuals 
Rscript src/seismophobia_eda.R --data_path=data/processed/train.csv --out_dir=visuals

# Run the modelling
python src/build_model.py --input_train_file_path="data/processed/train.csv" --input_test_file_path="data/processed/test.csv" --output_visuals_path="visuals"

# Write the report
Rscript -e "rmarkdown::render('doc/seismophobia_report.Rmd', output_format='github_document')"

# Write out report so it visualizes nice on Github
Rscript -e "rmarkdown::render('doc/seismophobia_report.Rmd', output_file='index.html',output_dir='.', output_format='github_document')"
