# UWB LOS/NLOS Signal Classification and Distance Estimation

## 1. Project Overview
This project addresses precise indoor localization using Ultra-Wideband (UWB) wireless signals.
In indoor environments, obstacles such as walls and doors create Non-Line-of-Sight (NLOS) conditions,
which can introduce significant ranging errors.

The solution applies the 3D Process of Data Analytics to:
- Classify signals as LOS or NLOS for two dominant paths.
- Estimate measured range for those paths to improve localization accuracy.

## 2. The 3D Process Implementation

### A. Data Preparation (The 1st D)
The pipeline processes an empirical dataset with 42,000 samples from 7 indoor environments.

- Data cleaning: Consolidates 7 CSV files and strips whitespace from headers.
- CIR normalization: Normalizes Channel Impulse Response (CIR) values by `RXPACC`.
- Feature extraction: Derives `Path2_Amp` and `Path2_Delay` from raw CIR data.
- Feature importance: Uses Random Forest ranking to identify key predictors (for example, `RXPACC`, `CIR_PWR`).

### B. Data Mining (The 2nd D)
A supervised learning workflow is used with an 80:20 train-test split.

- Classification: Compares Random Forest and Multi-Layer Perceptron (MLP) for LOS/NLOS prediction.
- Regression: Trains distance estimators for `Measured_range` on Path 1 and Path 2.
- Path logic: Applies the project rule that Path 2 is treated as NLOS.

### C. Data Visualization (The 3rd D)
Performance indicators are visualized to validate model behavior.

- Confusion matrix for classification performance.
- Actual vs. Predicted scatter plots for regression quality.
- RMSE-based summary for Path 1 and Path 2 ranging error.

## 3. Key Results
| Metric | Path 1 (LOS/NLOS) | Path 2 (NLOS) |
|---|---:|---:|
| Classification Accuracy | 91.40% | Defined by Path 1 logic |
| Regression RMSE | 1.5027 m | 1.7823 m |

## 4. Project Structure
```
UWB-LOS-NLOS-Data-Set/
	code/
		CSC3105_Main_Pipeline.ipynb
	dataset/
		uwb_dataset_part1.csv
		uwb_dataset_part2.csv
		uwb_dataset_part3.csv
		uwb_dataset_part4.csv
		uwb_dataset_part5.csv
		uwb_dataset_part6.csv
		uwb_dataset_part7.csv
	LICENSE.txt
	overleaf_report.tex
	overleaf_report.pdf
	cover.pdf
	README.md
```

## 5. How to Run
### Clone the Repository
```bash
git clone https://github.com/jacobfss777/csc3105_data_project.git
cd csc3105_data_project
```

### Install Dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn numpy jupyter
```

### Run the Notebook
Open `code/CSC3105_Main_Pipeline.ipynb` in VS Code and run all cells in order.

The notebook expects the CSV files to remain in the `dataset/` folder with the current filenames.