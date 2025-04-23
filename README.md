# CISC5790 - Alzheimer's Disease Data Analysis

This repository contains a Jupyter Book that explores Alzheimer's disease data analysis using various machine learning techniques, including data retrieval, preprocessing, and multiple classification models.

## Repository Structure

```
CISC5790Project/
├── book/                 # Jupyter Book source files
│   ├── introduction.ipynb
│   ├── data_retrieval.ipynb
│   ├── create_oversampled.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── KNN.ipynb
│   ├── Decision Tree.ipynb
│   ├── RandomForest.ipynb
│   ├── stacking.ipynb
│   ├── _config.yml       # Book configuration
│   ├── _toc.yml          # Table of contents
│   └── requirements.txt  # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```  

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/simonedinato/CISC5790Project.git
   cd CISC5790Project
   ```
2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r book/requirements.txt
   ```

## Building the Jupyter Book

1. Navigate into the `book/` directory:
   ```bash
   cd book
   ```
2. Build the book without re-executing notebooks (execution is disabled):
   ```bash
   jupyter-book build .
   ```
3. After building, open the generated HTML:
   ```bash
   open _build/html/index.html
   ```

## Usage

- **Introduction**: Introduces the project, outlines Alzheimer's disease background, dataset description, and research objectives.
- **Data Retrieval**: Demonstrates fetching and loading the raw Alzheimer's dataset (e.g., from ADNI), with initial exploratory checks.
- **Create Oversampled**: Shows how to balance the dataset using oversampling techniques like SMOTE or random oversampling to address class imbalance.
- **Data Preprocessing**: Covers data cleaning steps including missing value imputation, encoding categorical variables, feature scaling, and feature selection.
- **KNN**: Implements the k-Nearest Neighbors classifier, explores hyperparameter tuning, and evaluates performance using accuracy, precision, recall, and confusion matrix.
- **Decision Tree**: Builds a decision tree model, performs parameter tuning, visualizes the tree, and analyzes feature importance.
- **Random Forest**: Trains a random forest ensemble, tunes hyperparameters, and compares its performance against single-tree and KNN models.
- **Stacking**: Demonstrates a stacked ensemble approach by combining multiple base learners to improve overall predictive accuracy and robustness.

## Bibliography

All references are stored in `book/references.bib` and can be cited throughout the notebooks.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request on GitHub.

## License

Specify license details here. 