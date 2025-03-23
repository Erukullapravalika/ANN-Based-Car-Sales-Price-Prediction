# **Car Purchase Amount Prediction Using ANN**
    
### **Project Overview**
This project aims to predict the amount a customer will spend on a car using a deep learning model. The dataset contains various factors such as customer demographics, financial details, and additional synthetic features like advertising spend and promotions. The model uses an Artificial Neural Network (ANN) to make accurate predictions.

### **Problem Statement**
Businesses need to forecast customer spending on cars based on historical data. This prediction can help optimize marketing strategies and budget allocation. The goal is to:
  * Analyze different factors affecting car purchase amount.
  * Handle missing values and outliers effectively.
  * Apply feature scaling for better model performance.
  * Train an ANN model that accurately predicts car purchase amounts.
  * Evaluate model performance using suitable metrics.
  * Provide a structured implementation with clear documentation.
  
### **Dataset Details**
The dataset includes:
  * Customer Information: Gender, Age, Salary, Credit Card Debt, Net Worth
  * Financial Information: Advertising Spend, Promotions
  * Categorical Features: Customer Segmentation
  * Target Variable: Car Purchase Amount
  
### **Data Preprocessing**
  1.	Handling Missing Values: Missing values were filled appropriately.
  2.	Outlier Removal: Used percentile-based filtering to remove extreme values.
  3.	Feature Engineering: 
       -Created new features like Advertising Spend, Promotions, and Customer Segment.
       -Applied one-hot encoding for categorical variables.
  4.	Feature Scaling: 
       -Used MinMaxScaler for scaling input features and the target variable.

### **Model Architecture**
A deep learning model was implemented using Keras with the following architecture:
  * Input Layer: Accepts numerical input features.
  * Hidden Layers: 
      o	Fully connected (Dense) layers with Leaky ReLU activation.
      o	Batch Normalization to improve training stability.
      o	Dropout Layers to prevent overfitting.
  * Output Layer: Single neuron predicting car purchase amount.

### **Hyperparameters**:
  * Optimizer: Adam (learning rate = 1e-3)
  * Loss Function: Mean Squared Error (MSE)
  * Epochs: 500 (with Early Stopping)
  * Batch Size: 64
  
### **Model Training & Evaluation**
The model was trained using:
  * Train-Test Split: 80% training, 20% testing.
  * Early Stopping:Stops training if validation loss doesn't improve for 50 epochs.
  
### **Performance Metrics**
  * Mean Absolute Error (MAE): Measures average prediction error.
  * R2 Score: Indicates how well predictions fit actual values.
  
### **Final Model Performance**:
  * Mean Absolute Error: 1329.5076689015145
  * R2 Score: 0.9759168087450423 (indicating a very strong fit)
  
### **Results & Visualization**
  * Loss vs. Epochs: Plotted to show training convergence.
  * Actual vs. Predicted Values: Scatter plot comparing real and predicted amounts.
  * Ideal Fit Line: Helps visualize model accuracy.

### **Sample Prediction Function**
A function is provided to predict car purchase amounts for new customers. Example:
```
sample_data = X_test.iloc[0].values  # Taking the first test sample as input
predicted_amount = predict_car_price(sample_data)
print(f"Predicted Car Purchase Amount: {predicted_amount}")
```

### **Repository Structure**
```
|--Sales Prediction
      |-- Car_Sales_Forecasting  # Jupyter Notebook with Full Implementation
      |-- car_purchasing.csv  # Dataset (if applicable)
      |-- README.md  # Project Documentation
```
    
### **Conclusion**
This ANN-based solution effectively predicts car purchase amounts with high accuracy. The structured repository ensures ease of understanding and reproducibility. Further improvements could include testing other ML models like Random Forest and XGBoost for comparison.
________________________________________
### ***Next Steps***:
  * Deploy the model using Flask or FastAPI for real-time predictions.
  * Explore feature importance analysis to understand key influencers of purchase behavior.

