# CO₂ Emissions Prediction

Predict CO₂ emissions of vehicles using **linear regression** with multiple features. This project demonstrates both a **from-scratch gradient descent implementation** and **scikit-learn's LinearRegression**, allowing a direct comparison of performance.

---

## Getting Started

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd "CO2 Emission Prediction"
   ```

2. Install required packages:
   ```bash
   pip install numpy pandas matplotlib scikit-learn seaborn jupyter
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

---

## Dataset

- **Source:** `CO2 Emissions_Canada.csv`  
- **Features used:**
  - Engine Size (L)  
  - Cylinders  
  - Fuel Consumption City (L/100 km)  
  - Fuel Consumption Hwy (L/100 km)  
  - Fuel Consumption Comb (L/100 km)  
- **Target:** CO₂ Emissions (g/km)

---

## Methodology

1. **Data Preprocessing**
   - Selected numeric features relevant to CO₂ emissions.
   - Standardized features using Z-score normalization.
   - Split dataset into 80% training and 20% validation.
   - Shuffled data to avoid ordering bias.

2. **Gradient Descent (From Scratch)**
   - Implemented linear regression with **gradient descent** optimization.
   - Cost function: Mean Squared Error (MSE).
   - Computed gradients and updated weights iteratively.
   - Monitored cost convergence over iterations.

3. **Scikit-learn Linear Regression**
   - Trained `LinearRegression` model on the same standardized features.
   - Compared weights, bias, and prediction performance with custom gradient descent.

---

## Model Comparison

| Feature                          | Gradient Descent Weight | scikit-learn Weight |
|----------------------------------|-----------------------|-------------------|
| Engine Size (L)                  | 8.08                  | 7.22              |
| Cylinders                         | 11.99                 | 12.36             |
| Fuel Consumption City (L/100 km) | 14.43                 | 0.65              |
| Fuel Consumption Hwy (L/100 km)  | 10.07                 | 0.95              |
| Fuel Consumption Comb (L/100 km) | 13.17                 | 36.20             |
| **Bias**                          | 250.47                | 250.38            |

**Observations:**
- Both models achieve almost identical predictive performance.
- Differences in weights arise from feature correlations and differences in algorithm approach (iterative vs analytical solution).

---

## Evaluation Metrics

| Model                  | MSE      | R² Score |
|------------------------|----------|----------|
| Gradient Descent       | 388.36   | 0.891    |
| scikit-learn LinearReg | 388.75   | 0.891    |

- **Mean Squared Error (MSE):** Average squared difference between predicted and actual values.  
- **R² Score:** Proportion of variance explained by the model (closer to 1 → better fit).

---

## Visualizations

The notebook includes several visualizations:

1. **Feature Distribution Analysis:**
   - Scatter plots showing the relationship between each feature and CO₂ emissions
   - Helps identify correlations and data patterns

2. **Training Progress:**
   - Cost history during gradient descent iterations
   - Weight evolution over training iterations
   - Demonstrates convergence behavior

3. **Model Predictions Comparison:**
   - Actual vs. Predicted CO₂ Emissions scatter plot
   - Blue points → Gradient Descent predictions
   - Green points → scikit-learn predictions
   - Red dashed line → Ideal predictions (y = x)

---

## Project Structure

```
CO2 Emission Prediction/
│
├── main.ipynb                      # Jupyter notebook with complete implementation
├── CO2 Emissions_Canada.csv        # Dataset
└── README.md                       # Project documentation
```

---

## Key Learnings

- **Gradient Descent Implementation:** Understanding the mathematics behind linear regression optimization
- **Feature Scaling:** Importance of standardization for gradient descent convergence
- **Model Comparison:** Validating custom implementations against industry-standard libraries
- **Performance Metrics:** Using MSE and R² score for model evaluation

---

## Future Enhancements

- Implement regularization techniques (Ridge, Lasso)
- Explore polynomial features for non-linear relationships
- Add cross-validation for more robust evaluation
- Experiment with other regression algorithms (Random Forest, XGBoost)
- Create interactive visualizations using Plotly

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Author

Created as a machine learning demonstration project for CO₂ emissions prediction.
