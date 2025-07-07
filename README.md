# Soybean Futures Price Predictor

This is a Streamlit web app that uses machine learning models to forecast next-day soybean futures prices.  
The app downloads historical soybean futures closing prices, creates features, and lets you train either a Random Forest or XGBoost regression model with customizable hyperparameters.

---

## Features

- Uses **Yahoo Finance** data for soybean futures (symbol: `ZS=F`) from **January 1, 2022** to present.
- Automatically creates lag features (previous day’s and two days’ closing prices) and moving averages (3-day and 7-day).
- Allows user to select model type and tune:
  - Number of trees
  - Maximum tree depth
  - Train/test split percentage
- Shows model performance with RMSE metric.
- Visualizes actual vs predicted soybean prices.
- Displays feature importance from the trained model.
- Includes raw data viewer for quick inspection.

---

## How to run

1. Clone the repo or copy `app.py`.

2. (Recommended) Create a virtual environment and activate it:

```bash
python3 -m venv env
source env/bin/activate  # macOS/Linux
# or
env\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app locally:

```bash

streamlit run app.py
```

5. Use the sidebar controls to select model and tune parameters, then click Train Model.

## Notes

- Data is fetched dynamically from Yahoo Finance on app startup.

- The app caches data to avoid re-downloading.

- Training can take a few seconds depending on your parameters.

- Requires internet connection to download data.

- Tested on Python 3.13.3 and macOS.
