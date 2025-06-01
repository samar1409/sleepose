import streamlit as st
import numpy as np
import pandas as pd

def generate_sensor_data():
    np.random.seed(42)
    raw_stretch = np.random.normal(850, 25, 100)
    smoothed_stretch = np.convolve(raw_stretch, np.ones(5)/5, mode='valid')
    return raw_stretch, smoothed_stretch

def display_sensor_data():
    raw_stretch, smoothed_stretch = generate_sensor_data()
    
    st.header("Sensor Data Visualization")
    st.line_chart({"Raw Signal": raw_stretch, "Smoothed Signal": smoothed_stretch})
    
    st.markdown("**Edge Processing:** Moving average filter (5-sample window) applied in real time to reduce noise.")