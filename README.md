# Boston Housing ML + Flask (Beginner Tutorial)

> Note: The original Boston Housing dataset has known ethical issues and was removed from scikit‑learn. 
This tutorial uses a **small sample CSV** for practice. For better learning, replace `data/boston_sample.csv`
with a full dataset that includes these columns:

CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV

## Steps
1) Create & activate a virtual environment
```
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```
2) Install dependencies
```
pip install -r requirements.txt
```
3) Train & save the model
```
python train_and_save.py
```
This creates `models/model.joblib`

4) Run the Flask app
```
python app.py
```
Then open http://127.0.0.1:5000

## Files
- `train_and_save.py` — trains a Linear Regression model in a Pipeline and saves it.
- `app.py` — loads the saved model and serves a form for real‑time predictions.
- `templates/` — HTML templates for the form and result pages.
- `data/boston_sample.csv` — tiny demo dataset; replace with a full dataset for better results.