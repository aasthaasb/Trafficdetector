# -*- coding: utf-8 -*-
"""Random_Forest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hJvpvvPooaglNW4zWuUt-Dc0g_txmJcm

###Random Forest
"""

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=74)

# Fit the model using the training data
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)  # Calculate RMSE
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Model Metrics:")
print("Mean Squared Error:", mse_rf)
print("Root Mean Squared Error:", rmse_rf)
print("Mean Absolute Error:", mae_rf)
print("R-squared Score:", r2_rf)

# Define the parameter distribution for Random Forest
rf_param_dist = {
    'n_estimators': randint(100, 1000),  # Number of trees in the forest
    'max_depth': [None] + list(randint(1, 50).rvs(10)),  # Maximum depth of the tree
    'min_samples_split': randint(2, 20),  # Minimum number of samples required to split a node
    'min_samples_leaf': randint(1, 20)  # Minimum number of samples required at each leaf node
}

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=74)

# Initialize RandomizedSearchCV for Random Forest
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_param_dist, n_iter=100, cv=5, scoring='r2', random_state=74)

# Fit RandomizedSearchCV to find the best parameters for Random Forest
random_search_rf.fit(X_train_scaled, y_train)

# Get the best parameters for Random Forest
best_params_rf = random_search_rf.best_params_
print("Best Parameters for Random Forest:", best_params_rf)

# Use the best parameters to initialize the Random Forest model
best_rf_model = RandomForestRegressor(**best_params_rf, random_state=74)

# Fit the model using the training data
best_rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_rf = best_rf_model.predict(X_test_scaled)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)  # Calculate RMSE
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Model Metrics with Best Parameters (Randomized Search):")
print("Mean Squared Error:", mse_rf)
print("Root Mean Squared Error:", rmse_rf)
print("Mean Absolute Error:", mae_rf)
print("R-squared Score:", r2_rf)
