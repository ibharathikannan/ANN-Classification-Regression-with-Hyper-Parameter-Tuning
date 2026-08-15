import streamlit as st

# Must be the first Streamlit command in the script.
st.set_page_config(page_title="ANN Predictions", page_icon="\U0001F9E0")

# st.navigation/st.Page (not the classic pages/ auto-discovery folder) - lets us
# set explicit sidebar labels and icons instead of Streamlit deriving them from
# filenames (which is why the nav used to just say "app").
churn_page = st.Page("views/churn_prediction.py", title="Churn Prediction", icon="\U0001F4C9")
salary_page = st.Page("views/salary_regression.py", title="Salary Regression", icon="\U0001F4B0")

pg = st.navigation([churn_page, salary_page])
pg.run()
