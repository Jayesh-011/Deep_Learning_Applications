import streamlit as st

st.title("  by Jayesh Mahendra Patil")

name = st.text_input("Enter your name")

if name:
    st.success(f"Welcome {name}")