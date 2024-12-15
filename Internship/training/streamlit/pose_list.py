import pandas as pd
import streamlit as st  # Import Streamlit

# Load the class names dynamically from the Excel file
df = pd.read_excel("yoga_pose_classes.xlsx")
pose_list = df["Pose Name"].tolist()

# Sidebar for pose selection
selected_pose = st.sidebar.selectbox("Select a Pose", pose_list)

st.write(f"You selected: {selected_pose}")
