conda create -p venv python==3.11 -y
conda activate venv
pip install -r requirements.txt
pip install ipykernel
streamlit run app.py