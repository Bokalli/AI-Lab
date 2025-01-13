# Traffic Flow Prediction Using Neural Networks

This project aims to predict traffic flow patterns by integrating traffic and weather data using a neural network. The approach leverages Python and popular machine learning libraries to process data, engineer features, and build a predictive model.

---

## **Objectives**
1. Analyze traffic and weather data to identify trends.
2. Develop a model to predict traffic flow using historical traffic and weather information.
3. Visualize the relationships between weather conditions and traffic patterns.

---

## **Technologies Used**
- Python 3.8 or higher
- Libraries:
  - `NumPy`: Numerical computations.
  - `Pandas`: Data manipulation and cleaning.
  - `Matplotlib`: Data visualization.
  - `Scikit-learn`: Preprocessing and evaluation.
  - `TensorFlow/Keras`: Neural network implementation.

---

## **Steps Implemented**
1. **Data Preprocessing**:
   - **Traffic Data Simulation**: Created a dataset with timestamps and random traffic flow values.
   - **Weather Data Simulation**: Generated temperature and humidity data corresponding to traffic timestamps.
   - **Feature Engineering**:
     - Extracted time-based features (e.g., hour, day of the week).
     - Created lag features to include historical traffic flow data.
   - **Normalization**: Scaled numerical features to ensure effective model training.

2. **Model Development**:
   - Built a neural network with the following architecture:
     - Input layer for all features.
     - Two hidden layers using ReLU activation functions.
     - Output layer for traffic flow prediction.

3. **Model Training and Evaluation**:
   - Split data into training and testing sets.
   - Trained the model using `Mean Squared Error (MSE)` as the loss function.
   - Evaluated the model using `MSE` and `Mean Absolute Error (MAE)`.

4. **Visualization**:
   - Plotted predicted vs. actual traffic flow to evaluate model performance.
   - Created scatter plots to visualize the effect of temperature and humidity on traffic flow.

---

## **Results**
- The neural network demonstrated reasonable predictive performance, with low MSE and MAE values on test data.
- Visualizations revealed significant correlations:
  - Increased traffic flow was observed under specific temperature ranges.
  - Humidity showed a subtle but noticeable effect on traffic patterns.

---

## **Conclusion**
The project successfully demonstrated how integrating weather and traffic data could predict traffic patterns. While the neural network performed well, future improvements could include using real-world datasets and experimenting with advanced architectures like LSTMs for temporal dependencies.

---

## **Requirements**
- Python 3.8+
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
