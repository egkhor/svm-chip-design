# Support Vector Machine for Chip Design Defect Prediction

This open-source project showcases the **Support Vector Machine (SVM)** algorithm for machine learning, applied to predict defects in semiconductor chip designs. Using an SVM classifier with an RBF kernel, we analyze synthetic chip data (e.g., transistor count, defect rate) to achieve ~82% accuracy in defect detection, enhancing chip design efficiency.

## Features
- **SVM Model**: Maps chip metrics to high-dimensional spaces for accurate defect classification.
- **Synthetic Dataset**: 1,000 samples for prototyping.
- **Efficient**: Balances accuracy and compute for local execution.

## Why SVM vs. Decision Trees vs. LLMs?
- **SVM**: Excels with structured data, using non-linear boundaries (RBF kernel) for robust defect prediction, though less interpretable.
- **Decision Trees**: Simple and interpretable, ideal for small datasets, but prone to overfitting without pruning.
- **LLMs (e.g., DistilBERT)**: Suited for text analysis (e.g., design notes), but overkill for tabular data with high compute needs.

## Project Structure
```
svm-chip-design/
├── README.md
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── src/
│   └── svm_defect_prediction.py
└── data/
    ├── chip_defect_data.csv
    └── svm_metrics.txt
```

## Getting Started

### Prerequisites
- Python 3.8+
- Dependencies: `pip install -r requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/egkhor/svm-chip-design.git
   cd svm-chip-design
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python src/svm_defect_prediction.py
   ```

### Output
- Generates `chip_defect_data.csv` and `svm_metrics.txt` with model accuracy (~82%).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) to add datasets, models, or visualizations.

## License
MIT License. See [LICENSE](LICENSE).

## Contact
Open an Issue or join Discussions on [GitHub](https://github.com/egkhor/svm-chip-design).
