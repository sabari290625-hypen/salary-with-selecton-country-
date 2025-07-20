import sys
print("🚀 Running with Python:", sys.executable)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv("Salary_Data_10000 (1).csv")
df.dropna(inplace=True)

# Encode Education
df['Education Level'].replace(["Bachelor's Degree", "Master's Degree", "phD"], ["Bachelor's", "Master's", "PhD"], inplace=True)
df['Education Level'] = df['Education Level'].map({"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3})

# Gender and Age encoding (optional if not already in dataset)
if 'Gender' not in df.columns:
    df['Gender'] = np.random.choice(['Male', 'Female', 'Other'], size=len(df))
if 'Age' not in df.columns:
    df['Age'] = np.random.randint(22, 60, size=len(df))

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})

# Add missing columns
if 'Location' not in df.columns:
    df['Location'] = np.random.choice(['Urban', 'Rural', 'Suburban'], size=len(df))
    df['Job Type'] = np.random.choice(['Remote', 'Hybrid', 'Onsite'], size=len(df))

# Reduce and encode job titles
job_title_count = df['Job Title'].value_counts()
job_title_edited = job_title_count[job_title_count <= 25]
df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x)

# One-hot encoding
df = pd.get_dummies(df, columns=['Location', 'Industry', 'Job Title', 'Job Type'], drop_first=True)

# Final split
X = df.drop('Salary', axis=1)
y = df['Salary']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=20)
model.fit(x_train, y_train)

# 🌍 Country Map with Flags, Symbols, Rates
country_map = {
    "🇺🇸 USA": ("USD", "$", 1.00),
    "🇮🇳 India": ("INR", "₹", 86),
    "🇬🇧 UK": ("GBP", "£", 0.78),
    "🇩🇪 Germany": ("EUR", "€", 0.92),
    "🇫🇷 France": ("EUR", "€", 0.92),
    "🇮🇹 Italy": ("EUR", "€", 0.92),
    "🇪🇸 Spain": ("EUR", "€", 0.92),
    "🇳🇱 Netherlands": ("EUR", "€", 0.92),
    "🇧🇪 Belgium": ("EUR", "€", 0.92),
    "🇸🇪 Sweden": ("SEK", "kr", 10.6),
    "🇳🇴 Norway": ("NOK", "kr", 10.4),
    "🇨🇭 Switzerland": ("CHF", "CHF", 0.89),
    "🇩🇰 Denmark": ("DKK", "kr", 6.8),
    "🇵🇹 Portugal": ("EUR", "€", 0.92),
    "🇦🇹 Austria": ("EUR", "€", 0.92),
    "🇫🇮 Finland": ("EUR", "€", 0.92),
    "🇮🇪 Ireland": ("EUR", "€", 0.92),
    "🇨🇦 Canada": ("CAD", "C$", 1.36),
    "🇦🇺 Australia": ("AUD", "A$", 1.48),
    "🇳🇿 New Zealand": ("NZD", "NZ$", 1.66),
    "🇯🇵 Japan": ("JPY", "¥", 157.5),
    "🇨🇳 China": ("CNY", "¥", 7.3),
    "🇰🇷 South Korea": ("KRW", "₩", 1380),
    "🇸🇬 Singapore": ("SGD", "S$", 1.35),
    "🇲🇾 Malaysia": ("MYR", "RM", 4.7),
    "🇮🇩 Indonesia": ("IDR", "Rp", 16200),
    "🇵🇭 Philippines": ("PHP", "₱", 58.0),
    "🇻🇳 Vietnam": ("VND", "₫", 24500),
    "🇹🇭 Thailand": ("THB", "฿", 36.3),
    "🇹🇷 Turkey": ("TRY", "₺", 33.5),
    "🇮🇱 Israel": ("ILS", "₪", 3.6),
    "🇸🇦 Saudi Arabia": ("SAR", "﷼", 3.75),
    "🇦🇪 UAE": ("AED", "د.إ", 3.67),
    "🇶🇦 Qatar": ("QAR", "﷼", 3.64),
    "🇧🇭 Bahrain": ("BHD", "ب.د", 0.38),
    "🇰🇼 Kuwait": ("KWD", "د.ك", 0.31),
    "🇿🇦 South Africa": ("ZAR", "R", 18.2),
    "🇳🇬 Nigeria": ("NGN", "₦", 1600),
    "🇪🇬 Egypt": ("EGP", "E£", 48.5),
    "🇷🇺 Russia": ("RUB", "₽", 89.0),
    "🇧🇷 Brazil": ("BRL", "R$", 5.2),
    "🇦🇷 Argentina": ("ARS", "$", 920),
    "🇨🇱 Chile": ("CLP", "$", 950),
    "🇵🇪 Peru": ("PEN", "S/", 3.6),
    "🇨🇴 Colombia": ("COP", "$", 4000),
    "🇲🇽 Mexico": ("MXN", "$", 18.0),
    "🇵🇰 Pakistan": ("PKR", "₨", 278),
    "🇧🇩 Bangladesh": ("BDT", "৳", 118),
    "🇱🇰 Sri Lanka": ("LKR", "Rs", 304),
    "🇳🇵 Nepal": ("NPR", "Rs", 133),
}

# ----------------- Streamlit UI ------------------

st.title("💼 Global Salary Prediction App")

# Inputs
gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
age = st.slider("Your Age", 22, 60, 25)
experience_mode = st.radio("Experience Input Mode", ['Manual', 'Automatic'])

if experience_mode == 'Manual':
    experience = st.slider("Years of Experience", 0, 40, 3)
    if experience > (age - 22):
        st.error(f"🚫 Invalid input: At age {age}, maximum experience is {age - 22} years.")
        st.stop()
else:
    experience = max(0, age - 22)
    st.info(f"🧠 Experience automatically calculated as: {experience} years (Age - 22)")

education = st.selectbox("Education Level", ['High School', "Bachelor's", "Master's", "PhD"])
location = st.selectbox("Location", ['Urban', 'Rural', 'Suburban'])
industry = st.selectbox("Industry", ['Tech', 'Finance', 'Healthcare', 'Education'])
job_type = st.selectbox("Job Type", ['Remote', 'Hybrid', 'Onsite'])
job_title = st.selectbox("Job Title", sorted([col.replace("Job Title_", "") for col in X.columns if "Job Title_" in col] + ['Others']))

formatted_options = [f"{k}\n💱 {v[0]} ({v[1]})" for k, v in country_map.items()]
selected = st.selectbox("🌍 Country You Want to Work In", formatted_options)
country_display = selected.split('\n')[0]
currency_code, currency_symbol, conversion_rate = country_map[country_display]

# Encode input
input_data = {
    'Education Level': {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}[education],
    'Years of Experience': experience,
    'Gender': {'Male': 0, 'Female': 1, 'Other': 2}[gender],
    'Age': age
}

# One-hot encoding
for col in X.columns:
    if "Location_" in col:
        input_data[col] = 1 if col == f'Location_{location}' else 0
    if "Industry_" in col:
        input_data[col] = 1 if col == f'Industry_{industry}' else 0
    if "Job Title_" in col:
        input_data[col] = 1 if col == f'Job Title_{job_title}' else 0
    if "Job Type_" in col:
        input_data[col] = 1 if col == f'Job Type_{job_type}' else 0

input_df = pd.DataFrame([input_data])

# Predict
predicted_salary_usd = model.predict(input_df)[0]
converted_salary = predicted_salary_usd * conversion_rate
salary_in_inr = predicted_salary_usd * 86

# Output
st.subheader("📈 Predicted Salary:")
st.success(f"{currency_symbol} {converted_salary:,.2f} ({country_display})")
st.markdown(f"💰 Equivalent Salary in 🇮🇳 INR: ₹ {salary_in_inr:,.2f}")

# Feature importance
if st.checkbox("Show Feature Importances"):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))