# 🚀 Freelance Job Success Prediction & Bid Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Predict your chances of winning freelance projects and get intelligent bidding recommendations powered by Machine Learning.**

[📊 Model Results](#-model-performance) · [🚀 Quick Start](#-quick-start) · [📁 Project Structure](#-project-structure) · [🧪 Testing](#-testing)

</div>

---

## 📌 Overview

This end-to-end ML project helps freelancers make **data-driven decisions** when applying for projects. Using a trained Random Forest classifier, it predicts the probability of winning a project and provides **actionable bidding recommendations** based on that probability.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Success Prediction** | Predict winning probability for any freelance job |
| 💡 **Bid Recommendations** | Get strategic advice — Apply, Caution, or Avoid |
| 🌐 **Web Interface** | User-friendly Streamlit app with real-time results |
| 📦 **Batch Mode** | Analyze multiple jobs simultaneously |
| 📊 **Model Comparison** | 4 ML models trained and benchmarked |
| 🧪 **Full Test Suite** | 38 unit tests with 100% pass rate |

---

## 📈 Model Performance

| Model | F1 Score | Accuracy | ROC-AUC | Threshold |
|-------|----------|----------|---------|-----------|
| ✅ **Random Forest** | **0.8764** | **89.9%** | **0.9610** | **0.40** |
| Logistic Regression | 0.8753 | 89.8% | 0.9652 | 0.46 |
| Gradient Boosting | 0.8692 | 89.2% | 0.9627 | 0.33 |
| Decision Tree | 0.8532 | 88.4% | 0.9386 | 0.55 |

> **Best Model:** Random Forest Classifier — F1 Score: **0.8764** ✅ (Target: ≥ 0.75)

---

## 🎯 Recommendation Logic

| Success Probability | Level | Recommendation | Strategy |
|---------------------|-------|----------------|----------|
| **≥ 70%** | 🟢 High | ✅ Apply Immediately | High Priority Bid |
| **40% – 69%** | 🟡 Medium | ⚠️ Apply with Caution | Optimized Bid |
| **< 40%** | 🔴 Low | ❌ Avoid Applying | Preserve Resources |

---

## 📁 Project Structure

```
Freelance-Job-Success-Prediction/
│
├── 📂 app/
│   └── app.py                         # Streamlit web application
│
├── 📂 data/
│   ├── raw_dataset.csv                # Original generated dataset (5,100 rows)
│   ├── cleaned_freelance_jobs.csv     # Preprocessed dataset
│   └── validation_visualizations.png # EDA plots
│
├── 📂 models/
│   ├── best_model.pkl                 # Best model (Random Forest)
│   ├── random_forest.pkl
│   ├── logistic_regression.pkl
│   ├── gradient_boosting.pkl
│   ├── decision_tree.pkl
│   ├── scaler.pkl                     # StandardScaler
│   └── best_threshold.pkl             # Optimal classification threshold
│
├── 📂 notebooks/
│   ├── data_preparation.ipynb         # Phase 1: Data Cleaning
│   ├── eda.ipynb                      # Phase 2: Exploratory Data Analysis
│   └── model_training.ipynb           # Phase 3: Model Training & Evaluation
│
├── 📂 src/
│   ├── recommendation_engine.py       # Core prediction & recommendation logic
│   └── generate_dataset.py            # Synthetic dataset generator
│
├── 📂 tests/
│   ├── test_app.py                    # App unit tests
│   ├── test_recommendation_engine.py  # Engine unit tests
│   ├── conftest.py                    # Shared pytest fixtures
│   └── run_tests.py                   # Test runner script
│
├── 📂 plots/                          # All generated visualizations (.png)
├── 📂 reports/
│   └── Business_Insights.pdf          # Business insights report
│
├── generate_dataset.py                # Root-level dataset generator
├── requirements.txt                   # Python dependencies
├── pytest.ini                         # Pytest configuration
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Library | Version |
|----------|---------|---------|
| Language | Python | 3.10.11 |
| Data Processing | Pandas, NumPy | 2.0.3, 1.24.3 |
| Machine Learning | Scikit-learn | 1.3.0 |
| Visualization | Matplotlib, Seaborn, Plotly | 3.7.2, 0.12.2, 5.14.1 |
| Web App | Streamlit | 1.25.0 |
| Imbalanced Learning | Imbalanced-learn | 0.11.0 |
| Serialization | Joblib | 1.3.1 |
| Testing | Pytest, pytest-cov | 7.4.2, 4.1.0 |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/hamzamunirml/Freelance-Job-Success-Prediction-Bid-Recommendation-System.git
cd Freelance-Job-Success-Prediction-Bid-Recommendation-System
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate Dataset & Train Models

```bash
# Step 1: Generate dataset
python generate_dataset.py

# Step 2: Run notebooks in order
jupyter notebook
# → notebooks/data_preparation.ipynb
# → notebooks/eda.ipynb
# → notebooks/model_training.ipynb
```

### 5. Launch the App

```bash
streamlit run app/app.py
```

Open **http://localhost:8501** in your browser.

---

## 📊 How to Use

### Option A — Streamlit Web App *(Recommended)*

```bash
streamlit run app/app.py
```

Fill in the job details form and get instant predictions with visual probability gauge and recommendations.

### Option B — Recommendation Engine (CLI)

```bash
python src/recommendation_engine.py
```

### Option C — Jupyter Notebooks

```bash
jupyter notebook
```

Open any notebook in the `notebooks/` folder for step-by-step walkthrough.

---

## 📝 Sample Inputs & Expected Outputs

### ✅ High Probability (≥ 70%) — Apply Immediately

```
Project Budget:        $5,000
Client Rating:         4.8 ⭐
Existing Proposals:    5
Experience:            10 years
Proposal Quality:      9/10
Freelancer Rating:     4.7 ⭐
Jobs Completed:        200
Category:              Data Science
```
**Output:** `✅ High Probability — Apply Immediately`

---

### ⚠️ Medium Probability (40–69%) — Apply with Caution

```
Project Budget:        $25,000
Client Rating:         3.5 ⭐
Existing Proposals:    40
Experience:            6 years
Proposal Quality:      6/10
Freelancer Rating:     3.5 ⭐
Jobs Completed:        80
Category:              Design
```
**Output:** `⚠️ Medium Probability — Apply with Caution`

---

### ❌ Low Probability (< 40%) — Avoid Applying

```
Project Budget:        $50,000
Client Rating:         1.5 ⭐
Existing Proposals:    95
Experience:            2 years
Proposal Quality:      3/10
Freelancer Rating:     2.1 ⭐
Jobs Completed:        10
Category:              Writing
```
**Output:** `❌ Low Probability — Avoid Applying`

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_recommendation_engine.py -v
```

### Test Results

| Category | Tests | Passed | Status |
|----------|-------|--------|--------|
| App Mappings | 6 | 6 | ✅ |
| App Calculations | 5 | 5 | ✅ |
| App Data Validation | 5 | 5 | ✅ |
| Recommendation Engine | 15 | 15 | ✅ |
| Integration Tests | 2 | 2 | ✅ |
| Performance Tests | 2 | 2 | ✅ |
| **Total** | **38** | **38** | **✅ 100%** |

---

## 📊 Business Insights (from EDA)

1. **Proposal Quality is #1 Predictor** — Top quality (9–10) achieves 55.8% success vs. 18.5% for poor proposals (+0.42 correlation)
2. **Less Competition = Higher Wins** — Jobs with <10 proposals: 52.5% success; 50+ proposals: 22.5% success
3. **Experience Pays Off** — Expert freelancers (10+ yrs): 48.5% win rate vs. Entry (0–2 yrs): 22.5%
4. **Client Rating Matters** — Excellent clients (4–5★): 47.5% win rate; Poor clients (1–2★): 24.8%
5. **Category Selection Impacts Success** — Data Science leads (42.5%) vs. Writing lags (28.3%)

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Model file not found` | Run `model_training.ipynb` first |
| `Streamlit not found` | Run `pip install streamlit==1.25.0` |
| `Feature mismatch error` | Ensure all 20 features are provided |
| `scikit-learn version error` | Pin to `scikit-learn==1.3.0` |

```bash
# Debug commands
python --version                          # Check Python 3.10.x
pip list | grep scikit                    # Verify scikit-learn version
python -c "import joblib; m=joblib.load('models/best_model.pkl'); print(type(m).__name__)"
```

---

## 🏆 Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Data Preparation & Cleaning | ✅ Complete |
| Phase 2 | Exploratory Data Analysis | ✅ Complete |
| Phase 3 | Model Training & Evaluation | ✅ Complete |
| Phase 4 | Recommendation Engine | ✅ Complete |
| Phase 5 | Streamlit Web App | ✅ Complete |
| Testing | 38 Unit Tests (100% Pass) | ✅ Complete |

---

## 🔮 Future Enhancements

- [ ] XGBoost / LightGBM model integration
- [ ] Deploy on Streamlit Cloud
- [ ] REST API with FastAPI
- [ ] Real freelance data integration (Upwork/Fiverr)
- [ ] User authentication & history tracking
- [ ] Mobile app development
- [ ] A/B testing for recommendation strategies

---

## 📞 Contact

**Hamza Munir**
- 🎓 B.S. Artificial Intelligence — KFUEIT, Rahim Yar Khan
- 💼 [LinkedIn](https://linkedin.com/in/hamzamunirml)
- 🐙 [GitHub](https://github.com/hamzamunirml)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Scikit-learn](https://scikit-learn.org/) — ML algorithms
- [Streamlit](https://streamlit.io/) — Web framework
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — Data processing
- [Pytest](https://pytest.org/) — Testing framework

---

<div align="center">

**⭐ Star this repo if you found it useful!**

Made with ❤️ by [Hamza Munir](https://github.com/hamzamunirml)

</div>
