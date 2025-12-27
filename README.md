# CO₂ Emissions Prediction

Predict CO₂ emissions of vehicles using **linear regression** with multiple features. This project demonstrates both a **from-scratch gradient descent implementation** and **scikit-learn's LinearRegression**, allowing a direct comparison of performance.

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

## Visualization

- **Validation Predictions vs Actual CO₂ Emissions:**
  - Blue points → Gradient Descent predictions
  - Green points → scikit-learn predictions
  - Red dashed line → Ideal predictions
