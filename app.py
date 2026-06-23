import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🎭 Simple Sentiment Analyzer")

user_input = st.text_area("Enter a review:")

if st.button("Analyze"):
    if user_input.strip() != "":
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        st.write(f"Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text.")
