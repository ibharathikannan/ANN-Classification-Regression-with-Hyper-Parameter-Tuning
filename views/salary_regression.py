import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model and preprocessing artifacts.
# Cached so this only runs once per session, not on every widget interaction
# (Streamlit re-runs the whole script top-to-bottom on every rerun otherwise).
@st.cache_resource
def load_artifacts():
    # compile=False: not needed for inference, and matches notebook 04's defensive
    # pattern (avoids a Keras-3 loss-deserialization bug that appears on newer
    # TensorFlow/Keras versions than this project's pinned 2.15.0).
    model = tf.keras.models.load_model('regression_model.h5', compile=False)

    # label_encoder_gender and onehot_encoder_geo are shared with the churn
    # prediction page - both are fit on Gender/Geography before the
    # classification/regression target split even happens, so they're identical
    # either way. Only the scaler differs (different feature columns per task),
    # so that one stays regression-specific.
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('regression_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, label_encoder_gender, onehot_encoder_geo, scaler


model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()


## streamlit app
st.title('Estimated Salary Prediction')

# User input
# Same fields as the churn prediction page, except 'EstimatedSalary' (now the
# target, not an input) and with 'Exited' added (it's an input feature here,
# not the target).
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited (previously churned)', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited],
})

# One-hot encode 'Geography'
# This encoder was pickled with sparse_output=False, so .transform() already
# returns a dense array - no .toarray() needed.
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Enforce the exact column order the scaler was fit on - see notebooks 04/05 for why
# this matters (a mismatched order fails loudly for a DataFrame, but silently and
# incorrectly for a plain array, so it's worth being explicit rather than relying
# on the concat above happening to produce the right order).
input_data = input_data[scaler.feature_names_in_]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict estimated salary
prediction = model.predict(input_data_scaled, verbose=0)
predicted_salary = prediction[0][0]

st.write(f'Predicted Estimated Salary: {predicted_salary:,.2f}')
