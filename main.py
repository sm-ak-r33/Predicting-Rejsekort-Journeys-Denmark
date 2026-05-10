import os

# --- Daily pipeline ---
os.system("python pipeline/ingest_daily.py")
os.system("python pipeline/preprocessing.py")
os.system("python pipeline/autoarima.py")
os.system("python pipeline/selected_arima.py")
os.system("python pipeline/prophet_model.py")
os.system("python pipeline/BiLSTM.py")

# --- Monthly pipeline ---
os.system("python pipeline/ingest_monthly.py")
os.system("python pipeline/monthly_forecast.py")
