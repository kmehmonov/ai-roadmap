1. **Arithmetic mean**
   $$
   \bar{X} = \frac{\sum_{i=1}^n x_i}{n}
   $$

---

2. **Geometric Mean**: Used for data involving growth rates:
   $$
   \text{Geometric Mean} = \left( \prod_{i=1}^n x_i \right)^{\frac{1}{n}}
   $$

---

3. **Harmonic Mean**: Used for rates and ratios:
   $$
   \text{Harmonic Mean} = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}
   $$

---

4. **Variance**

$$
\text{Var}(X) = \sigma_X^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})^2 = 
$$

---

5. **Covariance**

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})
$$

---

6. **Correlation**

$$
\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

Where \( \sigma_X \) and \( \sigma_Y \) are the standard deviations of \( X \) and \( Y \).

---

7. **Determination (R²):**
   $$
   R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
   $$

Where:

- \( y_i \) is the actual value,
- \( \hat{y}\_i \) is the predicted value,
- \( \bar{y} \) is the mean of the actual values.

---

8. **Mean Squared Error (MSE):**
   $$
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   $$

---

9. **Mean Absolute Error (MAE):**
   $$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$

---

10. **Slope (\(b_1\)):**
$$
b_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})}{\sum_{i=1}^{n} (x_i - \bar{X})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
$$

11. **Intercept (\(b_0\)):**
$$
b_0 = \bar{Y} - b_1 \bar{X}
$$

