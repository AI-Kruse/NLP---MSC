# NLP---MSC
The coding and retrevial of relevant literature have been conducted 31 January 2026. By Sander Marko Kruse
Repository is used for a research article by applying NLP techniques and LDA on screened literature.
The coding, figures, CSV files can be found in the repository along with the literature being analysed.


# PDF Full-Text NLP - LDA Pipeline

This repository contains a complete NLP and Topic Modeling pipeline designed to analyze a corpus of research articles. The workflow transitions from raw PDF extraction to optimized LDA models and UMAP visualizations.

## Project Structure
- `NLP_LDA_Analysis_Full-text.py`: The main Python script for full text analyzing.
- `NLP_LDA_Analysis_Abstracts.py`: The main Python script for abstract analyzing.
- `Results_Abstracts/`: Visualizations and metrics from abstract-level analysis.
- `Results_FullText/`: Visualizations and metrics from full-text article analysis.
- **Releases**: Download the 56MB `.zip` file containing the 98 PDF articles from the [Releases tab](https://github.com/AI-Kruse/NLP---MSC/releases/download/v0.1/Samling.av.dokumenter.zip).

## How to Reproduce the Analysis
To run this code successfully, follow these specific steps to ensure the file paths match the script's logic:

1. **Environment**: Ensure you have Python installed with the following libraries: `sklearn`, `pandas`, `numpy`, `matplotlib`, `umap`, `fitz`, `wordcloud`, `NLTK`, `rispy`, `seaborn`
2. **File Placement (CRITICAL)**: 
   - Download the `.ris` files (bibliographic data) from this repository.
   - **Place all .ris files in the same folder as the `.py` script.** The code uses relative paths and expects these files to be in the root directory.
3. **Data**: Download the article ZIP from the **Releases** section and extract the PDFs into a folder where `NLP_LDA_Analysis_Full-text.py` is saved.
4. **Execution**: Run the script in Spyder or your preferred IDE.

## Methodology
- **Pre-processing**: Custom stopword removal and Bigram generation.
- **Optimization**: Manual Grid Search for the best `Alpha`, `Beta`, and `n_components`.
- **Visualization**: UMAP projections to map the thematic landscape of the corpus.

## Citation
If you use this code in your research, please cite it as:
> S. M. Kruse, "NLP - LDA Pipeline: Analysis of Abstracts vs Full-text corpus," GitHub, 2026. [Source code]. Available: [https://github.com/AI-Kruse/NLP---MSC]
