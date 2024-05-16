
# Splitting the data into features (X) and No. of vehicles (y)
X = data.drop('Vehicles', axis=1)
y = data['Vehicles']

# Splitting the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=74)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using the KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)

# Fit the model
knn.fit(X_train_scaled, y_train)



# Predicting the target values
y_pred = knn.predict(X_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)
