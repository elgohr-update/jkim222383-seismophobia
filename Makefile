## Seismophobia Analysis
## Group 11
## 2020/12/01

## This makefile will run the full analysis from data download all the way to writing the final report.


all : doc/seismophobia_report.md doc/seismophobia_report.html

# Get data from fivethirtyeight url
data/raw/earthquake.csv : src/download_csv.py
	python src/download_csv.py 'https://raw.githubusercontent.com/fivethirtyeight/data/master/san-andreas/earthquake_data.csv' data/raw/earthquake.csv

# Rename columns, add target column
data/processed/train.csv data/processed/test.csv : data/raw/earthquake.csv src/pre_process_seismophobia.R
	Rscript src/pre_process_seismophobia.R --input_path=data/raw/earthquake.csv --out_dir=data/processed

# Build eda visuals 
visuals/feature_distributions.png visuals/feature_distributions_across_response.png visuals/target_distribution.png : data/processed/train.csv data/processed/test.csv src/seismophobia_eda.R 
	Rscript src/seismophobia_eda.R --data_path=data/processed/train.csv --out_dir=visuals

# Run the modelling
visuals/classifier_results_table.png visuals/confusion_matrix_DummyClassifier.png visuals/confusion_matrix_RandomForestClassifier.png visuals/confusion_matrix_LogisticRegression.png  visuals/roc_auc_curve_DummyClassifier.png visuals/roc_auc_curve_RandomForestClassifier.png visuals/roc_auc_curve_LogisticRegression.png visuals/shap_summary_plot_LogisticRegression.png visuals/shap_summary_plot_RandomForestClassifier.png  :  data/processed/train.csv data/processed/test.csv src/build_model.py
	python src/build_model.py --input_train_file_path="data/processed/train.csv" --input_test_file_path="data/processed/test.csv" --output_visuals_path="visuals"

# Write the report
doc/seismophobia_report.md doc/seismophobia_report.html : visuals/classifier_results_table.png visuals/confusion_matrix_DummyClassifier.png visuals/confusion_matrix_RandomForestClassifier.png visuals/confusion_matrix_LogisticRegression.png  visuals/roc_auc_curve_DummyClassifier.png visuals/roc_auc_curve_RandomForestClassifier.png visuals/roc_auc_curve_LogisticRegression.png visuals/shap_summary_plot_LogisticRegression.png visuals/shap_summary_plot_RandomForestClassifier.png  visuals/feature_distributions.png visuals/feature_distributions_across_response.png visuals/target_distribution.png doc/seismophobia_report.Rmd
	Rscript -e "rmarkdown::render('doc/seismophobia_report.Rmd', output_format='github_document')"

clean :
	rm -f data/raw/earthquake.csv
	rm -f data/processed/*
	rm -f visuals/*
	rm -f doc/seismophobia_report.md doc/seismophobia_report.html


