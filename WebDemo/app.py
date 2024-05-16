import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import datetime
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

data=pd.read_csv("traffic.csv")

data['DateTime'] = pd.to_datetime(data['DateTime'])

# Extract and assign components of the datetime to new columns
data['Year'] = data['DateTime'].dt.year
data['Month'] = data['DateTime'].dt.month
data['DayOfMonth'] = data['DateTime'].dt.day
data['Hour'] = data['DateTime'].dt.hour
data['Minute'] = data['DateTime'].dt.minute
data['Second'] = data['DateTime'].dt.second

data['WeekDay'] = data['DateTime'].dt.weekday + 1
data = data.drop(['ID', 'DateTime'], axis=1)

data.tail()
data = data.drop([ 'Minute', 'Second'], axis=1)
X = data.drop('Vehicles', axis=1)  # This drops the target column from the data to create the features dataset
y = data['Vehicles']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Assuming other imports and data setup are done earlier in the code...
input_fields = [
    gr.Number(label="Junction"),
    gr.Number(label="Year"),
    gr.Number(label="Month"),
    gr.Number(label="DayOfMonth"),
    gr.Number(label="Hour"),
    gr.Number(label="WeekDay")
]

# knn = KNeighborsRegressor(n_neighbors=3)
# knn.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=74)
rf.fit(X_train_scaled, y_train)

# Gradient Boosting
# gb = GradientBoostingRegressor(n_estimators=100, random_state=74)
# gb.fit(X_train_scaled, y_train)

# # Decision Tree
# dt = DecisionTreeRegressor(random_state=74)
# dt.fit(X_train_scaled, y_train)

# SVM
# svm = SVR()
# svm.fit(X_train_scaled, y_train)

# Linear Regression
# lr = LinearRegression()
# lr.fit(X_train_scaled, y_train)

# Predict function now includes all models
def predict_vehicle(junction, year, month, day_of_month, hour, weekday):
    try:
        # Create a DataFrame for the features
        features = pd.DataFrame([[junction, year, month, day_of_month, hour, weekday]],
                                columns=X_train.columns)
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make predictions using all models
        # prediction_knn = knn.predict(features_scaled)[0]
        prediction_rf = rf.predict(features_scaled)[0]
        # prediction_gb = gb.predict(features_scaled)[0]
        # prediction_dt = dt.predict(features_scaled)[0]
        # prediction_svm = svm.predict(features_scaled)[0]
        # prediction_lr = lr.predict(features_scaled)[0]

        # Return all predictions
        return ( np.round(prediction_rf))
    except Exception as e:
        return f"An error occurred: {e}"

# Define the output fields for all models
output_fields = [
    # gr.Textbox(label="Predicted Vehicles (KNN)"),
    gr.Textbox(label="Predicted Vehicles (Random Forest)"),
    # gr.Textbox(label="Predicted Vehicles (Gradient Boosting)"),
    # gr.Textbox(label="Predicted Vehicles (Decision Tree)"),
    # gr.Textbox(label="Predicted Vehicles (SVM)"),
    # gr.Textbox(label="Predicted Vehicles (Linear Regression)")
]

# Create the Gradio interface with the function and the updated outputs
iface = gr.Interface(
    fn=predict_vehicle,
    inputs=input_fields,
    outputs=output_fields,
    title="Vehicle Prediction Model",
    description="Enter the features to predict the vehicle count using multiple models"
)

# Launch the Gradio interface
iface.launch()


