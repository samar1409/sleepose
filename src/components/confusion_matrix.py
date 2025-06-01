import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred):
    labels = ["Bad Sleep", "Good Sleep"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    st.pyplot(fig)

def generate_fake_data():
    # Simulated true and predicted labels
    y_true = np.random.choice([0, 1], size=100)
    y_pred = np.random.choice([0, 1], size=100)
    return y_true, y_pred

def display_confusion_matrix():
    st.header("Simulated Confusion Matrix")
    y_true, y_pred = generate_fake_data()
    plot_confusion_matrix(y_true, y_pred)