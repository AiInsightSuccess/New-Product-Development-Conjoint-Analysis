# Website Page Instructions

Use the text below on the webpage where users download the conjoint analysis
folder.

## Choice-Based Conjoint Analysis Notebook

Download the complete folder below to run the example choice-based conjoint
analysis on your own computer. The folder includes a Jupyter notebook, sample
CSV files, and example output files.

### Download

Download the ZIP file:

`website_download_package.zip`

After downloading, unzip the file before opening the notebook.

### What You Need

For basic users, we recommend installing Anaconda. Anaconda includes Python,
Jupyter Notebook, and the main data analysis packages used in this workbook.

- Download Anaconda: https://www.anaconda.com/download
- Anaconda installation guide:
  https://www.anaconda.com/docs/getting-started/anaconda/install
- Jupyter Notebook installation/help: https://jupyter.org/install

If you already use Python, install the required packages with:

```bash
pip install -r requirements.txt
```

### How to Run the Notebook

1. Download and unzip `website_download_package.zip`.
2. Install Anaconda if Python/Jupyter Notebook is not already installed.
3. Open Anaconda Navigator.
4. Click **Launch** under Jupyter Notebook.
5. In the Jupyter browser window, open the unzipped folder.
6. Open `Choice_Based_Conjoint_Analysis_Workbook.ipynb`.
7. Run the notebook cells from top to bottom.

### What Files Are Included

- `Choice_Based_Conjoint_Analysis_Workbook.ipynb`: the main notebook.
- `Feature_specs.csv`: feature names, types, level counts, and min/max values.
- `Feature_template.csv`: feature level values and labels.
- `Survey_response.csv`: sample survey responses.
- `requirements.txt`: package list for users not using Anaconda.
- `README.md`: detailed local instructions.
- `code/`: optional standalone Python scripts for advanced users.
- `outputs/`: example output files created by the notebook.

### What the Notebook Produces

The notebook creates:

- `fractional_factorial_design.csv`
- `levels_used.csv`
- `outputs/conjoint_model_dataset.csv`
- `outputs/conjoint_encoded_dataset.csv`
- `outputs/conjoint_model_metrics.csv`
- `outputs/conjoint_feature_significance.csv`
- `outputs/conjoint_coefficients.csv`
- `outputs/conjoint_best_combination.csv`
- `outputs/conjoint_top_combinations.csv`
- `outputs/conjoint_optimum_combination_long.csv`
- `outputs/conjoint_coefficients_by_feature.png`
- `outputs/conjoint_feature_p_values.png`
- `outputs/conjoint_top_combinations.png`

### What the Notebook Does

The notebook has four main parts:

1. **Import packages**  
   Loads common Python packages such as pandas, NumPy, SciPy, and Statsmodels.

2. **Set file locations**  
   Points to the input CSV files and output folder.

3. **Generate a fractional factorial design**  
   Reads the feature specification and template files, creates a design
   DataFrame, and saves `fractional_factorial_design.csv`.

4. **Run choice-based conjoint analysis**  
   Reads `Survey_response.csv` and `fractional_factorial_design.csv`, creates a
   model-ready dataset, fits logistic regression, prints pseudo R2 and p-values,
   lists significant features, prints coefficients for every feature level, and
   identifies the feature combination with the highest predicted probability.

5. **Create visual results**  
   Saves coefficient, feature p-value, and top-combination graphs in the
   `outputs/` folder, and prints the optimum combined feature-level result in a
   beginner-friendly table.

### Input File Format

`Feature_specs.csv` should contain:

- `Features`
- `Features_Type`
- `Level Number`
- `Minimum`
- `Maximum`

`Feature_template.csv` should contain one column per feature. Non-empty cells in
each column are treated as levels.

`Survey_response.csv` should contain:

- one row per participant
- one participant ID column, such as `Participant_ID`
- response columns named `run_id_1`, `run_id_2`, `run_id_3`, and so on
- binary response values, where `1` means selected/chosen and `0` means not
  selected/not chosen

### Common Problems

- **Jupyter does not open:** install Anaconda and launch Jupyter Notebook from
  Anaconda Navigator.
- **Package missing:** run `pip install -r requirements.txt`.
- **File not found:** keep the notebook and input CSV files in the same folder.
- **Level-count error:** check that `Feature_specs.csv` and
  `Feature_template.csv` have matching levels.
- **Response error:** check that response columns use `run_id_1`, `run_id_2`,
  etc., and contain only `0` and `1`.
