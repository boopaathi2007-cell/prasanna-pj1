# 🎓 Student Mark Prediction Machine Learning Project

An end-to-end Machine Learning system that predicts students' final academic marks ($G3$) based on demographics, study behaviors, attendance, and prior grades, utilizing the UCI Student Performance datasets (`student-mat.csv` and `student-por.csv`).

---

## 🌟 Key Features

1. **Multi-Model Regression Benchmarking**:
   - Compares **Ridge Regression, Linear Regression, Random Forest, Gradient Boosting, Extra Trees, and Support Vector Regressors (SVR)** using 5-fold cross-validation.
   - Evaluates $R^2$, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
   - Generates top performance:
     - **Mathematics**: Gradient Boosting ($R^2 = 0.828$, RMSE = $1.876$)
     - **Portuguese**: Ridge Regression ($R^2 = 0.849$, RMSE = $1.214$)
     - **Combined**: Random Forest ($R^2 = 0.829$, RMSE = $1.628$)

2. **Rich Grade Translation & Actionable Recommendations**:
   - Converts numeric continuous marks (0–20) into:
     - **Scaled Percentages** ($0 - 100\%$)
     - **Letter Grades** ($A, B, C, D, F$)
     - **Academic Standing** (*Distinction, Merit, Pass, At-Risk / Fail*)
   - Provides personalized recommendations on attendance, study hours, and academic trajectory.

3. **Interactive Streamlit Web Dashboard**:
   - **Student Mark Predictor**: 1-click presets (*Top Achiever, Average Student, At-Risk Student*) and customizable sliders.
   - **Interactive What-If Simulator**: Real-time simulation of score increases with higher study time or reduced absences.
   - **Leaderboard & Visualizations**: Model comparison charts, feature importance rankings, and actual vs. predicted plots.
   - **Dataset Analytics**: Mark distributions, passing rates, and raw data explorer.

4. **Fast Command-Line Interface (CLI)**:
   - Run single-line terminal predictions with custom arguments or student presets.

---

## 📁 Project Structure

```
d:/mlproject/
├── student-mat.csv             # Mathematics course dataset (395 students)
├── student-por.csv             # Portuguese course dataset (649 students)
├── student.txt                 # Dataset description & attributes
├── requirements.txt            # Python dependencies
├── models/                     # Trained Scikit-Learn pipelines & metadata
│   ├── best_model_math.pkl
│   ├── best_model_por.pkl
│   ├── best_model_combined.pkl
│   └── model_metadata.json
├── artifacts/                  # Visual plots & diagnostics
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── feature_importance.png
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion and train/test splits
│   ├── preprocessor.py         # Sklearn ColumnTransformer & encoding
│   ├── train.py                # Multi-model benchmarking script
│   └── predict.py              # Inference engine & grade translation
├── predict_cli.py              # Command-line prediction tool
└── app.py                      # Streamlit interactive web dashboard
```

---

## 🚀 Quickstart Guide

### 1. Installation
Install the project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train and Benchmark Models
Train all 6 regression algorithms on Math, Portuguese, and Combined datasets:
```bash
python src/train.py
```
This evaluates models, serializes the best pipelines to `models/`, and generates charts in `artifacts/`.

---

### 3. Command-Line Predictions (`predict_cli.py`)

#### Run with Preset Profiles:
```bash
# High-achieving student preset
python predict_cli.py --preset "Top Achiever"

# At-risk student preset
python predict_cli.py --preset "At-Risk"
```

#### Run with Custom Inputs:
```bash
# Mathematics course
python predict_cli.py --subject math --g1 15 --g2 16 --studytime 3 --absences 2

# Portuguese course
python predict_cli.py --subject por --g1 12 --g2 13 --studytime 2 --absences 4
```

---

### 4. Launch the Web Dashboard (`app.py`)
Run the interactive Streamlit application:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Model Evaluation Results

| Dataset | Top Algorithm | Test $R^2$ | RMSE | MAE | 5-Fold CV $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Mathematics** | Gradient Boosting | **0.8283** | **1.876** | 1.139 | 0.8886 |
| **Portuguese** | Ridge Regression | **0.8488** | **1.214** | 0.764 | 0.8362 |
| **Combined** | Random Forest | **0.8286** | **1.628** | 0.933 | 0.8561 |

---

## 🎓 Grading Rubric

| Mark ($G3$) | Percentage | Grade | Academic Standing | Status |
| :--- | :---: | :---: | :--- | :---: |
| **16.0 – 20.0** | 80% – 100% | **A** | Distinction / Outstanding | Pass |
| **14.0 – 15.9** | 70% – 79.9% | **B** | Merit / Very Good | Pass |
| **12.0 – 13.9** | 60% – 69.9% | **C** | Good / Satisfactory | Pass |
| **10.0 – 11.9** | 50% – 59.9% | **D** | Sufficient / Pass | Pass |
| **0.0 – 9.9** | 0% – 49.9% | **F** | Needs Intervention / Fail | Fail |
