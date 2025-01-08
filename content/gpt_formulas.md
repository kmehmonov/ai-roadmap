### **Activation Functions**  

**ReLU (Rectified Linear Unit)**  
\[
f(z) = \max(0, z)
\]  

**Leaky ReLU**  
\[
f(z) = \begin{cases} 
z & \text{if } z \geq 0 \\ 
\alpha z & \text{if } z < 0 
\end{cases}
\]  
Where \( \alpha \) is a small positive constant.  

**Tanh (Hyperbolic Tangent)**  
\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
\]  

---

### **Loss Functions**  

**Mean Squared Logarithmic Error (MSLE)**  
\[
\text{MSLE} = \frac{1}{n} \sum_{i=1}^n \left( \log(1 + y_i) - \log(1 + \hat{y}_i) \right)^2
\]  

**Hinge Loss**  
\[
\mathcal{L}(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})
\]  

**Categorical Cross-Entropy**  
\[
\mathcal{L} = - \sum_{i=1}^k y_i \log(\hat{p}_i)
\]  
Where \( k \) is the number of classes, \( y_i \) is a one-hot encoded vector, and \( \hat{p}_i \) is the predicted probability for class \( i \).  

---

### **Regularization**  

**L1 Regularization (Lasso Regression)**  
\[
\mathcal{J}(\theta) = J(\theta) + \lambda \|\theta\|_1 = J(\theta) + \lambda \sum_{j=1}^n |\theta_j|
\]  

**L2 Regularization (Ridge Regression)**  
\[
\mathcal{J}(\theta) = J(\theta) + \lambda \|\theta\|_2^2 = J(\theta) + \lambda \sum_{j=1}^n \theta_j^2
\]  

**Elastic Net Regularization**  
\[
\mathcal{J}(\theta) = J(\theta) + \lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2
\]  

---

### **Optimization Algorithms**  

**Stochastic Gradient Descent (SGD)**  
\[
\theta := \theta - \alpha \nabla_\theta J(\theta)
\]  

**Momentum Update**  
\[
v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta)
\]  
\[
\theta := \theta - \alpha v_t
\]  

**Adam Optimizer**  
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta)
\]  
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta))^2
\]  
\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\]  
\[
\theta := \theta - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\]  

---

### **Neural Networks**  

**Forward Propagation**  
\[
a^{[l]} = \sigma\left(W^{[l]} a^{[l-1]} + b^{[l]}\right)
\]  

**Backward Propagation (Gradient Computation)**  
\[
\frac{\partial J}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^\top
\]  
\[
\frac{\partial J}{\partial b^{[l]}} = \delta^{[l]}
\]  

---

### **Convolutional Neural Networks (CNNs)**  

**Convolution Operation**  
\[
s(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} x(i+m, j+n) \cdot w(m, n) + b
\]  
Where \( x \) is the input, \( w \) is the filter/kernel, \( b \) is the bias term, and \( k \) is the filter size.  

---

### **Evaluation Metrics**  

**Precision**  
\[
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
\]  

**Recall**  
\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\]  

**F1 Score**  
\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]  

**ROC AUC Score**  
\[
\text{AUC} = \int_{0}^{1} TPR \, d(FPR)
\]  


---

### **Support Vector Machines (SVM)**  

**Decision Boundary**  
\[
f(x) = w^\top x + b
\]  

**Hinge Loss for SVM**  
\[
\mathcal{L}(w, b) = \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i (w^\top x_i + b)) + \frac{\lambda}{2} \|w\|_2^2
\]  

---

### **Principal Component Analysis (PCA)**  

**Covariance Matrix**  
\[
\Sigma = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^\top
\]  

**Eigenvalue Decomposition**  
\[
\Sigma v = \lambda v
\]  

Where \( \lambda \) represents the eigenvalues and \( v \) represents the eigenvectors.  

---

### **k-Nearest Neighbors (k-NN)**  

**Distance Metric: Euclidean Distance**  
\[
d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
\]  

**Distance Metric: Manhattan Distance**  
\[
d(x, y) = \sum_{i=1}^n |x_i - y_i|
\]  

---

### **Clustering Algorithms**  

#### **K-Means Clustering**  

**Centroid Update**  
\[
\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
\]  
Where \( C_k \) is the cluster of data points assigned to centroid \( k \).  

**Within-Cluster Sum of Squares (WCSS)**  
\[
\text{WCSS} = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|_2^2
\]  

#### **Gaussian Mixture Models (GMM)**  

**Gaussian Probability Density Function**  
\[
p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)
\]  

**Expectation Step (E-step)**  
\[
\gamma_{i,k} = \frac{\pi_k p(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j p(x_i | \mu_j, \Sigma_j)}
\]  

**Maximization Step (M-step)**  
\[
\pi_k = \frac{1}{n} \sum_{i=1}^n \gamma_{i,k}
\]  
\[
\mu_k = \frac{\sum_{i=1}^n \gamma_{i,k} x_i}{\sum_{i=1}^n \gamma_{i,k}}
\]  
\[
\Sigma_k = \frac{\sum_{i=1}^n \gamma_{i,k} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_{i=1}^n \gamma_{i,k}}
\]  

---

### **Reinforcement Learning**  

#### **Q-Learning**  

**Q-value Update Rule**  
\[
Q(s, a) := Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]  
Where:  
- \( Q(s, a) \): State-action value.  
- \( \alpha \): Learning rate.  
- \( \gamma \): Discount factor.  

#### **Policy Gradient**  

**Objective Function**  
\[
J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^T R_t \log \pi_\theta(a_t | s_t) \right]
\]  

**Gradient Update**  
\[
\theta := \theta + \alpha \nabla_\theta J(\theta)
\]  

---

### **Deep Learning Regularization**  

#### **Dropout Regularization**  

**Dropout Operation**  
\[
a_i^{(l)} = 
\begin{cases} 
0 & \text{with probability } p \\
\frac{a_i^{(l)}}{1-p} & \text{with probability } 1-p
\end{cases}
\]  

---

### **Time Series Analysis**  

**Autoregressive Model (AR)**  
\[
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \dots + \phi_p X_{t-p} + \epsilon_t
\]  

**Moving Average Model (MA)**  
\[
X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q}
\]  

**ARIMA Model**  
\[
X_t = \phi_1 X_{t-1} + \dots + \phi_p X_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \dots + \theta_q \epsilon_{t-q}
\]  

---

### **Bayesian Inference**  

**Bayes' Theorem**  
\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\]  

---

### **Information Theory**  

**Entropy**  
\[
H(X) = -\sum_{i=1}^n P(x_i) \log P(x_i)
\]  

**KL Divergence**  
\[
D_{KL}(P || Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)}
\]  

**Cross-Entropy Loss**  
\[
H(p, q) = -\sum_{i=1}^n p(x_i) \log q(x_i)
\]  

---

### **Natural Language Processing (NLP)**  

**TF-IDF (Term Frequency-Inverse Document Frequency)**  

- **Term Frequency (TF):**  
\[
\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
\]  

- **Inverse Document Frequency (IDF):**  
\[
\text{IDF}(t) = \log\left(\frac{\text{Total number of documents}}{\text{Number of documents containing term } t}\right)
\]  

- **TF-IDF:**  
\[
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)
\]  


---

### **1. Joint Probability**
\[
P(A, B) = P(A|B)P(B) = P(B|A)P(A)
\]
Where \(P(A, B)\) is the probability of events \(A\) and \(B\) happening together.

---

### **2. Marginal Probability**
\[
P(A) = \sum_B P(A, B)
\]
Where the probability of \(A\) is computed by summing over all possible values of \(B\).

---

### **3. Conditional Probability**
\[
P(A|B) = \frac{P(A, B)}{P(B)} \quad \text{(if } P(B) \neq 0\text{)}
\]

---

### **4. Bayes' Theorem**
\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\]
Where \(P(A)\) is the prior, \(P(B|A)\) is the likelihood, and \(P(B)\) is the evidence.

---

### **5. Chain Rule for Joint Probabilities**
\[
P(x_1, x_2, \dots, x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2) \dots P(x_n|x_1, x_2, \dots, x_{n-1})
\]

---

### **6. Gaussian Distribution**
\[
P(x) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
\]
Where:
- \(x\): random variable
- \(\mu\): mean
- \(\sigma^2\): variance

---

### **7. Multivariate Gaussian Distribution**
\[
P(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)
\]
Where:
- \(x\): \(n\)-dimensional vector
- \(\mu\): mean vector
- \(\Sigma\): covariance matrix

---

### **8. Hidden Markov Models (HMMs)**

#### **Forward Algorithm (Recursive Formula)**
\[
\alpha_t(i) = P(O_1, O_2, \dots, O_t, S_t = i) = \sum_{j=1}^N \alpha_{t-1}(j) a_{ji} b_i(O_t)
\]
Where:
- \(a_{ji}\): transition probability from state \(j\) to \(i\)
- \(b_i(O_t)\): emission probability of observation \(O_t\) in state \(i\)

#### **Backward Algorithm (Recursive Formula)**
\[
\beta_t(i) = P(O_{t+1}, O_{t+2}, \dots, O_T | S_t = i) = \sum_{j=1}^N a_{ij} b_j(O_{t+1}) \beta_{t+1}(j)
\]

#### **Viterbi Algorithm**
\[
\delta_t(i) = \max_{j} \left[\delta_{t-1}(j) a_{ji}\right] b_i(O_t)
\]

---

### **9. Expectation-Maximization (EM Algorithm)**

#### **E-Step (Expectation Step)**
Compute the posterior probabilities:
\[
\gamma_{i,k} = P(Z_k = 1 | x_i; \theta) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
\]

#### **M-Step (Maximization Step)**
Update the parameters:
\[
\pi_k = \frac{1}{n} \sum_{i=1}^n \gamma_{i,k}
\]
\[
\mu_k = \frac{\sum_{i=1}^n \gamma_{i,k} x_i}{\sum_{i=1}^n \gamma_{i,k}}
\]
\[
\Sigma_k = \frac{\sum_{i=1}^n \gamma_{i,k} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_{i=1}^n \gamma_{i,k}}
\]

---

### **10. Bayesian Networks**

#### **Joint Probability in Bayesian Networks**
\[
P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i | \text{Parents}(X_i))
\]

---

### **11. Naive Bayes Classifier**

#### **Prediction Formula**
\[
P(C|X) \propto P(C) \prod_{i=1}^n P(X_i|C)
\]
Where:
- \(P(C)\): Prior probability of class \(C\)
- \(P(X_i|C)\): Likelihood of feature \(X_i\) given class \(C\)

---

### **12. KL Divergence**
\[
D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
\]

---

### **13. Maximum Likelihood Estimation (MLE)**
\[
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n P(x_i | \theta)
\]
Alternatively (log-likelihood):
\[
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log P(x_i | \theta)
\]

---

### **14. Maximum A Posteriori (MAP)**
\[
\hat{\theta}_{\text{MAP}} = \arg\max_\theta P(\theta | x) = \arg\max_\theta P(x | \theta)P(\theta)
\]

---

### **15. Dirichlet Distribution**
\[
P(\theta | \alpha) = \frac{1}{B(\alpha)} \prod_{i=1}^K \theta_i^{\alpha_i - 1}
\]
Where:
- \(\alpha = (\alpha_1, \alpha_2, \dots, \alpha_K)\): Concentration parameters
- \(B(\alpha)\): Beta function (normalization constant)

---

### **16. Latent Dirichlet Allocation (LDA)**

#### **Topic-Word Distribution**
\[
\phi_k = P(w | z = k)
\]

#### **Document-Topic Distribution**
\[
\theta_d = P(z | d)
\]

#### **Posterior Probability of Topic Assignment**
\[
P(z_{d,n} = k | w_{d,n}, \theta, \phi) \propto \theta_{d,k} \phi_{k, w_{d,n}}
\]

