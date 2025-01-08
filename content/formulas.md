**Arithmetic mean**
   $$
   \bar{X} = \frac{\sum_{i=1}^n x_i}{n}
   $$

---

**Geometric Mean**: Used for data involving growth rates:
   $$
   \text{Geometric Mean} = \left( \prod_{i=1}^n x_i \right)^{\frac{1}{n}}
   $$

---

**Harmonic Mean**: Used for rates and ratios:
   $$
   \text{Harmonic Mean} = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}
   $$

---

**Variance**

$$
\text{Var}(X) = \sigma_X^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})^2 
$$

---

**Covariance**

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})
$$

---

**Correlation**

$$
\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

Where \( \sigma_X \) and \( \sigma_Y \) are the standard deviations of \( X \) and \( Y \).

---

**Determination (R²):**
   $$
   R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
   $$

Where:

- \( y_i \) is the actual value,
- \( \hat{y}\_i \) is the predicted value,
- \( \bar{y} \) is the mean of the actual values.

---

**Mean Squared Error (MSE):**
   $$
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   $$

---

**Mean Absolute Error (MAE):**
   $$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$

---

**Slope (\(b_1\)):**
$$
b_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{X})(y_i - \bar{Y})}{\sum_{i=1}^{n} (x_i - \bar{X})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}
$$

---

**Intercept (\(b_0\)):**
$$
b_0 = \bar{Y} - b_1 \bar{X}
$$

---

**L1 norm** of a vector, also known as the **Manhattan norm** or **Taxicab norm**.
For a vector \( \mathbf{v} = [v_1, v_2, \dots, v_n] \), the L1 norm is given by:

$$
\|\mathbf{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|
$$

---

**L2 norm** of a vector, also called the **Euclidean norm**.
For a vector \( \mathbf{v} = [v_1, v_2, \dots, v_n] \), the L2 norm is calculated as:

$$
\|\mathbf{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}
$$

---

**Sigmoid Function**  
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

For a vector \( \mathbf{z} = [z_1, z_2, \ldots, z_n] \).

---

**Step Function**  
\[
f(z) =
\begin{cases} 
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
\]

---

**Softmax Function**  
\[
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
\]

---

**Linear Regression**
$$
\hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n
$$


**Matrix Form**  

\[
\hat{y} = X\theta
\]

**\(\theta\)**: Parameter Vector  
\[
\theta =
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_n
\end{bmatrix}
\]  

**\(X\)**: Feature Matrix
\[
X =
\begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix}
\]  

---
**Cost function for Linear Regression**
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left( \hat{y}_i - y_i \right)^2
$$


**Derivative**
$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}
$$

**Matrix Form**
$$
\nabla_\theta J(\theta) = \frac{1}{m} X^\top (X\theta - y)
$$

---
**Gradient Descent and the Cost Function:**

$$
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
$$

---

**Logistic Regression**
\[
\hat{y} = \frac{1}{1 + e^{-\left( \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n \right)}}
\]

**Matrix Form**
\[
\hat{y} = \sigma(X\theta) = \frac{1}{1 + e^{-X\theta}}
\]  

---

**Log Loss for single instance**
$$
c(\theta) = \begin{cases} 
-\log(\hat{p}) & \text{if } y = 1 \\
-\log(1-\hat{p}) & \text{if } y = 0 
\end{cases}
$$


**Log Loss** / **Logarithmic Loss** / **Binary Cross-Entropy Loss**
$$
\mathcal{J}(\theta) = -\frac{1}{m}\sum_{i=1}^m\left[y_i\log(\hat{p}_i) + (1 - y_i)\log(1 - \hat{p}_i) \right]
$$

**Derivative**
$$
\frac{\partial}{\partial \theta_j}J(\theta) = \frac{1}{m}\sum_{i=1}^m(\sigma(\theta^Tx_i) - y_i)x_i
$$

**Matrix Form**
\[
\nabla_\theta J(\theta) = \frac{1}{m} X^\top \left( \sigma(X\theta) - y \right)
\]

---



