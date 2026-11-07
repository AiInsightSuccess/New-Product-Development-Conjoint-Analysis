# Download and Run the Choice-Based Conjoint Analysis Workbook

Use this page to download a complete example folder for generating a fractional
factorial design and analyzing choice-based conjoint survey responses.

## What Is Included

The download folder includes:

- `Choice_Based_Conjoint_Analysis_Workbook.ipynb`: the Jupyter notebook you run.
- `Feature_specs.csv`: sample feature list, feature types, number of levels, and
  optional minimum/maximum values.
- `Feature_template.csv`: sample levels for each feature.
- `Survey_response.csv`: sample survey responses.
- `requirements.txt`: list of Python packages used by the notebook.
- `outputs/`: folder where the notebook saves model results.

## Before You Start

You need Python and Jupyter Notebook. The easiest option for beginners is
Anaconda because it installs Python, Jupyter, and common data-science packages
together.

Recommended setup:

1. Download Anaconda:
   https://www.anaconda.com/download
2. Follow Anaconda installation instructions:
   https://www.anaconda.com/docs/getting-started/anaconda/install
3. Learn how to install or launch Jupyter Notebook:
   https://jupyter.org/install

## Step-by-Step Instructions

1. Download the conjoint analysis folder from this webpage.
2. Unzip the folder.
3. Open Anaconda Navigator.
4. Launch Jupyter Notebook.
5. In Jupyter, open the downloaded folder.
6. Open `Choice_Based_Conjoint_Analysis_Workbook.ipynb`.
7. Run the notebook cells from top to bottom.
8. Review the printed results and the CSV files saved in the `outputs/` folder.

## What the Notebook Does

The notebook has four main parts:

1. Imports common Python packages.
2. Sets input and output file locations.
3. Generates a fractional factorial design from `Feature_specs.csv` and
   `Feature_template.csv`.
4. Fits a logistic regression choice-based conjoint model using
   `Survey_response.csv` and the generated design.

## Main Outputs

After running the notebook, the most useful files are:

- `fractional_factorial_design.csv`: generated fractional factorial design.
- `levels_used.csv`: levels used for each feature.
- `outputs/conjoint_model_dataset.csv`: analysis-ready dataset with outcome and
  feature levels.
- `outputs/conjoint_model_metrics.csv`: pseudo R2 and model p-value.
- `outputs/conjoint_feature_significance.csv`: p-values for each feature.
- `outputs/conjoint_coefficients.csv`: coefficient, p-value, and odds ratio for
  every feature level.
- `outputs/conjoint_best_combination.csv`: feature combination with the highest
  predicted probability.
- `outputs/conjoint_optimum_combination_long.csv`: the best combination in an
  easy-to-read feature/level table.
- `outputs/conjoint_coefficients_by_feature.png`: visual coefficient plot for
  each feature and level.
- `outputs/conjoint_feature_p_values.png`: visual p-value plot for features.
- `outputs/conjoint_top_combinations.png`: visual plot of the highest predicted
  combinations.

## Notes for Beginners

- Keep the notebook and the three input CSV files in the same folder.
- Do not rename the input files unless you also update the file-location cell in
  the notebook.
- The survey response file should have one row per participant and columns named
  `run_id_1`, `run_id_2`, and so on.
- A p-value below `0.05` is commonly treated as statistically significant.
- The R2 shown is McFadden pseudo R2, which is commonly used for logistic
  regression. It is not interpreted the same way as ordinary linear-regression
  R2.

## Troubleshooting

If the notebook cannot find a file, make sure all CSV files are in the same
folder as the notebook.

If a package is missing, open a terminal in the downloaded folder and run:

```bash
pip install -r requirements.txt
```

If Jupyter does not open, install or reinstall Anaconda and launch Jupyter
Notebook from Anaconda Navigator.
