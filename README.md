# 💼 Employee Salary Prediction

A Machine Learning web app that predicts employee salary using **Linear Regression**, built with **Streamlit**.

---

## 🚀 Live Demo
[Click here to view the app](#) <!-- Replace with your Streamlit Cloud URL -->

---

## 📌 Features
- Predicts salary based on Age, Experience, Education, Job Title & Gender
- Interactive sliders and dropdowns for input
- Clean, mobile-friendly UI
- Model trained with Scikit-learn Linear Regression

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML model |
| Streamlit | Web app UI |
| Joblib | Model serialization |

---

## 📂 Project Structure
```
salary_prediction/
│
├── app.py                  # Streamlit app (main UI)
├── train_model.py          # Script to train & save model artifacts
├── artifacts.pkl           # Saved model + encoders + scalers
├── Employee-salary-prediction.csv  # Dataset (add manually)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/salary-prediction.git
cd salary-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your dataset
Place `Employee-salary-prediction.csv` in the project root.

### 4. Train the model
```bash
python train_model.py
```
This generates `artifacts.pkl`.

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file** to `app.py`
5. Click **Deploy** 🚀

> **Note:** Upload `artifacts.pkl` to your repo (it's needed at runtime). Do **not** commit your CSV if it's private.

---

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| R² Score | ~85%+ |
| MAE | ~$8,000 |
| RMSE | ~$11,000 |

*(Exact values shown when you run `train_model.py`)*

---

## 👤 Author
**Rohit**  
B.Tech CSE (2023–2027) · Raj Kumar Goel Institute of Technology, Ghaziabad  
Data Science & Machine Learning Enthusiast
