import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# Streamlit UI setup
st.set_page_config(page_title="Stock Price Prediction", layout="centered")
st.title("📈 Stock Price Prediction")

# Load dataset
data = pd.read_csv("NFLX.csv")
st.success("✅ Dataset 'Netflix' successfully loaded.")

# Store date column separately (if present)
if 'Date' in data.columns:
    date_col = pd.to_datetime(data['Date'])  # convert to datetime
    data['Date'] = date_col  # store back for sorting

# Encode categorical columns
encoder = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = encoder.fit_transform(data[col])

# Model selection
model_name = st.selectbox("Select Prediction Model:", ["Decision Tree", "Linear Regression", "Random Forest"])

# Initialize model
if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Random Forest":
    model = RandomForestRegressor()
elif model_name == "Decision Tree":
    model = DecisionTreeRegressor()

# Sort by date to preserve time order
data = data.sort_values(by='Date')

# Features and target
X = data.drop(['Close', 'Date'], axis=1)
y = data['Close']
dates = data['Date']

# Split data
x_train, x_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, shuffle=False  # no shuffle to keep time order
)

# Train model
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)

# Show MSE
st.subheader("📊 Model Evaluation")
st.write(f"**Mean Squared Error (MSE):** `{mse:.4f}`")

# Convert to DataFrame for clean plotting
results_df = pd.DataFrame({
    'Date': dates_test,
    'Actual': y_test.values,
    'Predicted': y_pred
})
results_df.sort_values(by='Date', inplace=True)

# 📘 Plot Actual Stock Prices
st.subheader("📘 Actual Stock Prices")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(results_df['Date'], results_df['Actual'], label='Actual Price', color='blue')
ax1.set_title("Actual Stock Prices Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig1)

# 🔴 Plot Predicted Stock Prices
st.subheader("🔴 Predicted Stock Prices")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(results_df['Date'], results_df['Predicted'], label='Predicted Price', color='red')
ax2.set_title("Predicted Stock Prices Over Time")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
