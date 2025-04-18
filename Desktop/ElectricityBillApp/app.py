import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import pdfplumber
import io
import re
import pytesseract
import matplotlib.pyplot as plt
import seaborn as sns

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path

# Load the trained Random Forest model
with open("randomforest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to extract kWh usage from text
def extract_kwh_from_text(text):
    match = re.search(r'(\d+(\.\d+)?)\s*kWh', text, re.IGNORECASE)
    return float(match.group(1)) if match else None

# Streamlit UI
st.title("🔌 Electricity Bill Predictor")
st.write("Enter your monthly electricity consumption (in kWh) below.")

# Manual Input Section for kWh
st.subheader("🏠 Monthly Electricity Consumption (in kWh)")
kwh = st.number_input("Enter your monthly electricity consumption (in kWh)", min_value=0.0, step=0.1)

# Electricity Tariff Input
tariff_per_kwh = st.number_input("Electricity Tariff (₹/kWh)", min_value=0.0, step=0.1)

# Prediction Button
if st.button("Predict Bill Amount"):
    if kwh > 0 and tariff_per_kwh > 0:
        # Estimating the bill
        monthly_cost = kwh * tariff_per_kwh
        yearly_cost = monthly_cost * 12
        
        # Appliance-wise Breakdown (example, you can customize as needed)
        appliance_cost = {
            "AC": monthly_cost * 0.3,  # Example: AC uses 30% of the total bill
            "TV": monthly_cost * 0.1,  # Example: TV uses 10%
            "Others": monthly_cost * 0.6  # Rest of the appliances
        }

        # Cost Estimation
        st.success(f"💡 Predicted Monthly Bill Amount: ₹{monthly_cost:.2f}")
        st.write(f"📅 Yearly Cost Estimate: ₹{yearly_cost:.2f}")

        st.subheader("🔋 Appliance-Wise Cost Breakdown")
        for appliance, cost in appliance_cost.items():
            st.write(f"{appliance}: ₹{cost:.2f} ({(cost / monthly_cost) * 100:.2f}%)")

        # Efficiency Score
        efficiency_score = "Good" if monthly_cost < 1000 else "Average" if monthly_cost < 3000 else "High"
        st.write(f"📊 Energy Efficiency Score: {efficiency_score}")

        # Comparison with Similar Users (example)
        comparison_percentage = 15 if efficiency_score == "Good" else 5
        st.write(f"📊 Your home uses {comparison_percentage}% less energy than similar households.")

        # Personalized Energy-Saving Tips
        st.subheader("💡 Personalized Energy-Saving Tips")
        if monthly_cost > 500:
            st.write("🔋 **Tip:** Consider using your AC during off-peak hours (10 PM - 6 AM) to save energy.")
        if kwh > 500:
            st.write("🔋 **Tip:** Consider upgrading to energy-efficient appliances to reduce overall consumption.")
        if monthly_cost > 2000:
            st.write("🔋 **Tip:** Install a smart thermostat or timer to optimize your AC usage and reduce costs.")

        # Forecasting & Bill Prediction (example)
        next_month_prediction = monthly_cost * (1.05)  # Assuming 5% increase in usage
        st.write(f"📊 Your expected bill for next month is ₹{next_month_prediction:.2f}.")

        # Data Visualization
        st.subheader("📊 Data Visualization")
        labels = ["AC", "TV", "Others"]
        sizes = [appliance_cost["AC"], appliance_cost["TV"], appliance_cost["Others"]]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

        # Downloadable Report (CSV)
        report_data = {
            "Appliance": ["AC", "TV", "Others"],
            "Cost (₹)": [appliance_cost["AC"], appliance_cost["TV"], appliance_cost["Others"]],
            "Percentage": [(appliance_cost["AC"] / monthly_cost) * 100, (appliance_cost["TV"] / monthly_cost) * 100, (appliance_cost["Others"] / monthly_cost) * 100]
        }
        report_df = pd.DataFrame(report_data)
        st.download_button("Download Report", report_df.to_csv(), "report.csv", "text/csv")

    else:
        st.warning("⚠️ Please enter valid kWh and tariff values.")

# File Upload Section
st.markdown("---")
st.subheader("📁 Or Upload Your Electricity Bill")

uploaded_file = st.file_uploader("Choose a file (CSV, PDF, or image)", type=["csv", "pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.text("📄 Extracted Text:")
        st.text(text)

    elif "pdf" in file_type:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        st.text("📄 Extracted Text:")
        st.text(text)

    elif "csv" in file_type:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded CSV Data")
        st.dataframe(df)

    if "text" in locals():
        kwh = extract_kwh_from_text(text)
        if kwh:
            st.success(f"🔍 Detected usage: {kwh} kWh")
        else:
            st.warning("⚠️ Couldn't find kWh value. Try another file or enter manually.")
