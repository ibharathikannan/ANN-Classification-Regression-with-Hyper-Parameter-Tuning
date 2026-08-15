FROM python:3.11-slim

WORKDIR /app

# Deployment-only dependencies (streamlit, tensorflow-cpu, pandas, scikit-learn) -
# not the full dev requirements.txt, which also pulls in tensorboard/matplotlib/
# scikeras that are only needed for training/notebooks, not for serving the app.
# python:3.11-slim (not 3.12) because tensorflow==2.15.0 has no Python 3.12 wheel.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# App code: the classification page (app.py, the multi-page entrypoint) and the
# salary regression page (pages/1_Salary_Regression.py, Streamlit's native
# multi-page convention - anything in pages/ becomes a sidebar nav entry).
COPY app.py .
COPY pages ./pages

# Trained models + fitted preprocessing artifacts needed at inference time.
# label_encoder_gender.pkl / onehot_encoder_geo.pkl are shared by both pages;
# scaler.pkl / regression_scaler.pkl are task-specific (see notebook 06).
COPY model.h5 model.keras ./
COPY regression_model.h5 regression_model.keras ./
COPY label_encoder_gender.pkl onehot_encoder_geo.pkl scaler.pkl regression_scaler.pkl ./

EXPOSE 8501

# Streamlit's built-in health endpoint (not a generic "/" check, which Streamlit's
# own frontend doesn't respond to as a simple 200 the way a typical web server does).
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
