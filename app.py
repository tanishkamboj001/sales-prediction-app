import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("📈 Sales Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    if 'Advertising Budget' in data.columns and 'Sales' in data.columns:
        X = data[['Advertising Budget']]
        y = data['Sales']

        model = LinearRegression()
        model.fit(X, y)

        budget = st.slider("Select Advertising Budget", float(X.min()), float(X.max()))
        prediction = model.predict([[budget]])
        st.write(f"📊 Predicted Sales: **{prediction[0]:.2f}**")

        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(X, y, color='blue', label='Actual')
        ax.plot(X, model.predict(X), color='red', label='Regression Line')
        ax.set_xlabel("Advertising Budget")
        ax.set_ylabel("Sales")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("CSV must contain 'Advertising Budget' and 'Sales' columns.")
