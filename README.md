# Choice-Based Conjoint Analysis Workbook
[![DOI](https://zenodo.org/badge/1449815919.svg)](https://doi.org/10.5281/zenodo.24060441)

Free Python code and sample data from
[AI Insights for Success](https://www.aiinsightsuccess.com/blank-2) for designing
a choice-based conjoint experiment (D-optimal fractional factorial design) and
analyzing the survey responses with logistic regression.

It includes sample data, a beginner-friendly Jupyter notebook, and example
outputs.

## Get the Code

- **No Git?** Click the green **Code** button on this page, then
  **Download ZIP**, and unzip it.
- **Using Git:** `git clone https://github.com/AiInsightSuccess/New-Product-Development-Conjoint-Analysis.git`

Python 3.10 or newer is required (current Anaconda installs meet this).

## Recommended Beginner Setup

The easiest way to run the notebook is with Anaconda, because Anaconda installs
Python, Jupyter Notebook, and common data-analysis packages together.

Useful links:

- Download Anaconda: https://www.anaconda.com/download
- Anaconda installation guide:
  https://www.anaconda.com/docs/getting-started/anaconda/install
- Jupyter Notebook installation/help: https://jupyter.org/install

## Files in This Folder

- `Choice_Based_Conjoint_Analysis_Workbook.ipynb`: main notebook to run.
- `Feature_specs.csv`: sample feature names, feature types, level counts, and
  optional min/max values.
- `Feature_template.csv`: sample levels for each feature.
- `Survey_response.csv`: sample participant responses in wide format.
- `requirements.txt`: Python packages used by the notebook.
- `fractional_factorial_design.csv`, `levels_used.csv`: example design files
  created by the notebook.
- `code/`: optional standalone Python scripts for advanced users. The notebook
  does not import these scripts.
- `outputs/`: folder where conjoint-analysis results are saved (contains
  example results from the sample data).
- `docs/`: webpage copy drafts.

## Command-Line Use (Advanced)

From the repository folder:

```bash
python code/fractional_factorial_design.py --specs Feature_specs.csv --template Feature_template.csv --output fractional_factorial_design.csv --save-level-summary levels_used.csv
python code/run_conjoint_analysis.py --responses Survey_response.csv --output-dir outputs
```

## How to Run the Notebook

1. Download the repository (see *Get the Code* above).
2. Unzip the folder.
3. Open Anaconda Navigator.
4. Click `Launch` under Jupyter Notebook.
5. In Jupyter, browse to the downloaded folder.
6. Open `Choice_Based_Conjoint_Analysis_Workbook.ipynb`.
7. Run each notebook cell from top to bottom.
8. Review the printed results and the CSV files saved in `outputs/`.

If packages are missing, open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

## Input Files

Keep these files in the same folder as the notebook:

- `Feature_specs.csv`
- `Feature_template.csv`
- `Survey_response.csv`

The response file should have one row per participant and response columns named
`run_id_1`, `run_id_2`, `run_id_3`, and so on. Responses should be coded as `0`
or `1`.

## What the Notebook Does

The notebook:

1. Imports standard Python analysis packages.
2. Sets input and output file locations.
3. Generates a fractional factorial design from the feature CSV files.
4. Builds a model-ready dataset from the survey responses.
5. Fits a logistic regression choice-based conjoint model.
6. Prints R2, p-values, significant features, coefficients, and the best
   predicted feature combination.
7. Creates graphs for feature-level coefficients, feature p-values, and top
   predicted combinations.

## Main Outputs

After running the notebook, it creates:

- `fractional_factorial_design.csv`
- `levels_used.csv`
- `outputs/conjoint_model_dataset.csv`
- `outputs/conjoint_encoded_dataset.csv`
- `outputs/conjoint_model_metrics.csv`
- `outputs/conjoint_feature_significance.csv`
- `outputs/conjoint_significant_features.csv`
- `outputs/conjoint_coefficients.csv`
- `outputs/conjoint_significant_levels.csv`
- `outputs/conjoint_best_combination.csv`
- `outputs/conjoint_optimum_combination_long.csv`
- `outputs/conjoint_top_combinations.csv`
- `outputs/conjoint_coefficients_by_feature.png`
- `outputs/conjoint_feature_p_values.png`
- `outputs/conjoint_top_combinations.png`

## If Something Goes Wrong

- If Jupyter says a package is missing, run:

```bash
pip install -r requirements.txt
```

- If the notebook cannot find a CSV file, make sure the notebook and input CSVs
  are in the same folder.
- If the response file fails, check that response columns are named like
  `run_id_1`, `run_id_2`, and that responses are coded as `0` or `1`.
- If the feature files fail, check that the level counts in `Feature_specs.csv`
  match the non-empty values in `Feature_template.csv`.

The notebook is self-contained and does not import local custom packages.
