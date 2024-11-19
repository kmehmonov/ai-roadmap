# ai-roadmap


---

## Mathematics

### 1. **Linear Algebra**
   - **Core Concepts**: 
     - Vectors, matrices, and tensors
     - Matrix operations (addition, multiplication, transposition, inverse)
     - Eigenvalues and eigenvectors
     - Singular Value Decomposition (SVD)
   - **Applications**: Linear algebra forms the foundation for data representation (features are often vectors), dimensionality reduction techniques (e.g., PCA), and operations in neural networks.

   **Resources**:  
   - Gilbert Strang’s "Linear Algebra and Its Applications"  
   - 3Blue1Brown’s "Essence of Linear Algebra" YouTube series

### 2. **Calculus (Multivariable)**
   - **Core Concepts**: 
     - Derivatives and integrals
     - Partial derivatives
     - Gradient, Jacobian, and Hessian matrices
     - Chain rule (critical for backpropagation in neural networks)
   - **Applications**: Calculus is essential for understanding optimization techniques like gradient descent, where the objective is to minimize or maximize functions such as loss in machine learning models.

   **Resources**:  
   - MIT OpenCourseWare's Calculus courses  
   - Khan Academy’s Multivariable Calculus series

### 3. **Probability and Statistics**
   - **Core Concepts**: 
     - Probability distributions (Gaussian, Bernoulli, etc.)
     - Expectation, variance, covariance
     - Bayes' Theorem and conditional probability
     - Markov chains
     - Hypothesis testing, confidence intervals
   - **Applications**: Probabilistic models (e.g., Bayesian networks), generative models, and many machine learning algorithms like Naive Bayes, as well as evaluation metrics (e.g., precision, recall, ROC curve).

   **Resources**:  
   - "Probability and Statistics for Machine Learning" by José Unpingco  
   - StatQuest with Josh Starmer

### 4. **Optimization**
   - **Core Concepts**: 
     - Convex and non-convex optimization
     - Gradient-based optimization techniques (e.g., gradient descent, stochastic gradient descent)
     - Lagrange multipliers
   - **Applications**: Optimization is used to minimize the cost function in machine learning models, such as training neural networks by minimizing loss functions.

   **Resources**:  
   - "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe  
   - Coursera’s "Mathematics for Machine Learning: Multivariate Calculus" course

### 5. **Discrete Mathematics**
   - **Core Concepts**: 
     - Graph theory (important for social networks, knowledge graphs)
     - Combinatorics
     - Boolean algebra and logic
   - **Applications**: Algorithms like decision trees, search algorithms (used in AI planning), and graph-based methods in reinforcement learning or network-based AI tasks.

   **Resources**:  
   - "Discrete Mathematics and Its Applications" by Kenneth H. Rosen

### 6. **Linear/Logistic Regression and Applied Probability**
   - **Core Concepts**: 
     - Maximum Likelihood Estimation (MLE)
     - Logistic regression as a classifier
     - Regression analysis (least squares, gradient descent)
   - **Applications**: These are fundamental techniques for both classification and regression problems in supervised learning.

   **Resources**:  
   - "Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

### 7. **Information Theory**
   - **Core Concepts**: 
     - Entropy and cross-entropy
     - Kullback-Leibler divergence
     - Mutual information
   - **Applications**: Information theory is central to many AI techniques like decision trees (information gain), regularization (cross-entropy loss), and autoencoders.

   **Resources**:  
   - "Elements of Information Theory" by Thomas Cover and Joy Thomas

### 8. **Differential Equations**
   - **Core Concepts**: 
     - Ordinary Differential Equations (ODEs)
     - Partial Differential Equations (PDEs)
   - **Applications**: Deep learning models, such as neural ordinary differential equations, and physical modeling often rely on differential equations.

   **Resources**:  
   - MIT OpenCourseWare: Differential Equations

### 9. **Principal Component Analysis (PCA) and Matrix Factorization**
   - **Core Concepts**: 
     - Eigenvectors, Eigenvalues, Covariance matrix
     - Dimensionality reduction
     - SVD (Singular Value Decomposition)
   - **Applications**: Used in reducing dimensionality of large datasets, helping avoid overfitting, and improving computational efficiency.

   **Resources**:  
   - Coursera’s "Mathematics for Machine Learning: PCA" course

### 10. **Fourier Transforms and Signal Processing (Optional but Useful)**
   - **Core Concepts**: 
     - Fourier series and Fourier transforms
     - Signal filtering
   - **Applications**: Used in time-series analysis, image processing, and other tasks involving transformations between time and frequency domains.

   **Resources**:  
   - "The Fourier Transform and its Applications" on Coursera by Stanford University

### Overall Workflow for AI:
- **Data Representation**: Use linear algebra for representing and manipulating data.
- **Model Training**: Apply calculus (especially gradients) and optimization methods for learning.
- **Probabilistic Modeling**: Employ statistics and probability to quantify uncertainty and make predictions.
- **Inference and Decision Making**: Utilize graph theory, logic, and information theory for modeling complex AI systems like knowledge graphs and decision processes.

---

## Statistics

### 1. **Descriptive Statistics**
   - **Core Concepts**:
     - **Mean, Median, Mode**: Measures of central tendency that summarize data.
     - **Variance and Standard Deviation**: Measures of data spread or variability.
     - **Quartiles and Percentiles**: Dividing data into intervals, useful for understanding data distribution.
     - **Skewness and Kurtosis**: Measures that describe the shape of the data distribution.
   - **Applications**: Used to understand the basic structure and summary of datasets before applying machine learning models.

   **Resources**:  
   - Khan Academy: Introduction to Descriptive Statistics

### 2. **Probability Distributions**
   - **Core Concepts**:
     - **Discrete Distributions**: Such as Bernoulli, Binomial, and Poisson.
     - **Continuous Distributions**: Such as Normal (Gaussian), Uniform, and Exponential distributions.
     - **Probability Density Function (PDF)** and **Cumulative Distribution Function (CDF)**: Used to describe the likelihood of outcomes.
     - **Multivariate Distributions**: Understanding joint probabilities and conditional probabilities for multiple random variables.
   - **Applications**: Distributions are key for modeling uncertainty in data and predicting future outcomes, and they play a central role in hypothesis testing, regression models, and probabilistic modeling.

   **Resources**:  
   - "Probability and Statistics for Machine Learning" by José Unpingco  
   - StatQuest: Probability Distributions YouTube playlist

### 3. **Bayes’ Theorem and Conditional Probability**
   - **Core Concepts**:
     - **Conditional Probability**: The probability of an event given that another event has occurred.
     - **Bayes' Theorem**: Describes the relationship between conditional probabilities and prior beliefs.
     - **Prior, Likelihood, Posterior**: These are the components used in Bayesian statistics.
   - **Applications**: Crucial for Bayesian inference and probabilistic reasoning in AI, used in algorithms like Naive Bayes, Bayesian networks, and probabilistic programming.

   **Resources**:  
   - "Think Bayes" by Allen B. Downey (free book on Bayesian statistics)

### 4. **Inferential Statistics**
   - **Core Concepts**:
     - **Sampling**: Understanding how to draw representative samples from populations.
     - **Confidence Intervals**: Estimating the range within which a population parameter lies.
     - **Hypothesis Testing**: Determining whether a hypothesis about a dataset holds (e.g., t-test, chi-squared test, ANOVA).
     - **P-values**: The probability of observing data as extreme as the current data under the null hypothesis.
   - **Applications**: Used to make generalizations from sample data to populations, a critical step in validating machine learning models (e.g., A/B testing, statistical significance testing).

   **Resources**:  
   - Khan Academy: Inferential Statistics

### 5. **Regression Analysis**
   - **Core Concepts**:
     - **Linear Regression**: Modeling the relationship between a dependent variable and one or more independent variables.
     - **Logistic Regression**: Used for binary classification problems.
     - **Multivariate Regression**: Generalization of linear regression to multiple predictors.
     - **Coefficient of Determination (R²)**: Measures the proportion of variance explained by the model.
   - **Applications**: Linear regression is a basic machine learning algorithm for predicting continuous outcomes, while logistic regression is widely used for binary classification.

   **Resources**:  
   - "An Introduction to Statistical Learning" by James, Witten, Hastie, and Tibshirani

### 6. **Maximum Likelihood Estimation (MLE)**
   - **Core Concepts**:
     - **Likelihood Function**: The probability of observing the data given the model parameters.
     - **MLE**: A method of estimating model parameters that maximize the likelihood function.
   - **Applications**: Widely used for fitting statistical models, such as in logistic regression, Naive Bayes, and hidden Markov models.

   **Resources**:  
   - "Statistical Methods in Machine Learning" courses (Coursera or edX)

### 7. **Expectation-Maximization (EM) Algorithm**
   - **Core Concepts**:
     - **Expectation Step**: Calculates the expected value of the log-likelihood with respect to the conditional distribution of the latent variables.
     - **Maximization Step**: Maximizes the expected log-likelihood.
   - **Applications**: Used for clustering, like in Gaussian Mixture Models (GMMs), and in hidden Markov models for tasks like speech and handwriting recognition.

   **Resources**:  
   - "Pattern Recognition and Machine Learning" by Christopher M. Bishop

### 8. **Dimensionality Reduction Techniques**
   - **Core Concepts**:
     - **Principal Component Analysis (PCA)**: Reduces data dimensionality by finding the principal components.
     - **Linear Discriminant Analysis (LDA)**: Used for classification by maximizing the separation between multiple classes.
     - **t-SNE (t-distributed Stochastic Neighbor Embedding)**: Non-linear dimensionality reduction technique for visualization of high-dimensional datasets.
   - **Applications**: These techniques help with feature selection, reduce overfitting, and improve computational efficiency.

   **Resources**:  
   - Coursera’s "Mathematics for Machine Learning: PCA"  
   - "Pattern Recognition and Machine Learning" by Bishop

### 9. **Correlation and Covariance**
   - **Core Concepts**:
     - **Covariance**: Measures how two variables change together.
     - **Correlation**: Standardized form of covariance, measuring the strength and direction of a linear relationship between two variables.
   - **Applications**: Understanding relationships between features in datasets, often used in feature engineering and data preprocessing before applying machine learning models.

   **Resources**:  
   - Khan Academy: Correlation and Covariance

### 10. **Resampling Methods**
   - **Core Concepts**:
     - **Cross-Validation**: Splitting the dataset into multiple subsets for training and testing, used to evaluate model performance.
     - **Bootstrap**: Random sampling with replacement, used to estimate the distribution of statistics.
   - **Applications**: Resampling techniques are critical for model validation, ensuring that machine learning models generalize well to new data.

   **Resources**:  
   - "An Introduction to Statistical Learning" by James et al.

### 11. **Markov Chains and Hidden Markov Models (HMMs)**
   - **Core Concepts**:
     - **Markov Chains**: Models that describe systems which transition from one state to another based on certain probabilities.
     - **Hidden Markov Models (HMMs)**: Probabilistic models where the system's state is partially observable.
   - **Applications**: Widely used in AI tasks such as speech recognition, natural language processing, and time series analysis.

   **Resources**:  
   - "Hidden Markov Models and Bayesian Networks" by R. Ghahramani

### 12. **Bayesian Statistics**
   - **Core Concepts**:
     - **Prior and Posterior Distributions**: Prior knowledge updated with new evidence to form the posterior distribution.
     - **Conjugate Priors**: When the posterior distribution is the same type as the prior.
   - **Applications**: Bayesian methods are key in machine learning for making predictions in the presence of uncertainty, and in models like Bayesian networks and Gaussian processes.

   **Resources**:  
   - "Bayesian Data Analysis" by Andrew Gelman

### 13. **Chi-Squared Test and ANOVA (Analysis of Variance)**
   - **Core Concepts**:
     - **Chi-Squared Test**: Tests for independence between categorical variables.
     - **ANOVA**: Compares the means of three or more groups to see if at least one group is different.
   - **Applications**: Useful in hypothesis testing, particularly for evaluating categorical data and comparing model performance.

   **Resources**:  
   - Coursera’s Statistical Inference courses

### 14. **Bias-Variance Tradeoff**
   - **Core Concepts**:
     - **Bias**: Error due to simplifying assumptions in the model.
     - **Variance**: Error due to model sensitivity to small fluctuations in the training data.
     - **Overfitting and Underfitting**: Overfitting occurs when the model captures noise, while underfitting occurs when the model is too simple.
   - **Applications**: Fundamental for understanding model performance and generalization, critical in building robust machine learning models.

   **Resources**:  
   - "An Introduction to Statistical Learning" by James et al.


---

## Python

### 1. **Core Python Programming**
   - **Core Concepts**:
     - **Data Types**: Lists, dictionaries, tuples, sets, strings, and numbers.
     - **Control Flow**: Loops, conditional statements (if-else), list comprehensions.
     - **Functions**: Defining functions, lambda functions, higher-order functions.
     - **Object-Oriented Programming (OOP)**: Classes, objects, inheritance, polymorphism, encapsulation, and abstraction.
     - **Modules and Packages**: Importing and managing libraries, structuring code.
   - **Applications**: These core programming skills are necessary to structure machine learning workflows, build reusable code, and manage data efficiently.

   **Resources**:
   - "Automate the Boring Stuff with Python" by Al Sweigart (for beginners)
   - Python’s official documentation

### 2. **NumPy (Numerical Python)**
   - **Core Concepts**:
     - **Arrays**: Creating and manipulating multi-dimensional arrays.
     - **Broadcasting**: Automatic expansion of dimensions for operations.
     - **Linear Algebra**: Dot products, matrix operations, eigenvalues, eigenvectors.
     - **Mathematical Functions**: Statistical, trigonometric, and linear algebra functions.
   - **Applications**: NumPy is foundational for numerical computations in AI, enabling efficient data manipulation and serving as the basis for many higher-level libraries like pandas, TensorFlow, and PyTorch.

   **Resources**:
   - "Python Data Science Handbook" by Jake VanderPlas (covers NumPy in-depth)
   - NumPy official documentation

### 3. **Pandas (Data Manipulation and Analysis)**
   - **Core Concepts**:
     - **DataFrames and Series**: Data structures for handling tabular data.
     - **Data Cleaning**: Handling missing values, filtering, and selecting data.
     - **GroupBy, Merging, and Aggregation**: Summarizing and combining datasets.
     - **Time Series Analysis**: Working with time-indexed data.
   - **Applications**: Pandas is crucial for data preprocessing and exploratory data analysis (EDA), helping you clean, transform, and analyze data before building machine learning models.

   **Resources**:
   - "Python for Data Analysis" by Wes McKinney (creator of pandas)
   - Pandas official documentation

### 4. **Matplotlib and Seaborn (Data Visualization)**
   - **Core Concepts**:
     - **Plotting Basics**: Line plots, bar plots, histograms, scatter plots.
     - **Customizing Plots**: Labels, titles, legends, and colors.
     - **Seaborn**: High-level interface for drawing attractive and informative statistical graphics (e.g., pair plots, heatmaps).
   - **Applications**: Visualization is essential for understanding data distribution, feature relationships, and model performance. It also helps in presenting results effectively.

   **Resources**:
   - "Python Data Science Handbook" by Jake VanderPlas (for Matplotlib)
   - Seaborn official documentation

### 5. **Scikit-learn (Machine Learning Library)**
   - **Core Concepts**:
     - **Supervised Learning**: Classification (e.g., logistic regression, SVMs) and regression (e.g., linear regression, decision trees).
     - **Unsupervised Learning**: Clustering (e.g., k-means, DBSCAN) and dimensionality reduction (e.g., PCA).
     - **Model Selection**: Cross-validation, train-test split, grid search.
     - **Evaluation Metrics**: Accuracy, precision, recall, F1-score, ROC curve.
   - **Applications**: Scikit-learn provides easy-to-use implementations of many popular machine learning algorithms and tools for model selection and evaluation.

   **Resources**:
   - Scikit-learn official documentation
   - "Introduction to Machine Learning with Python" by Andreas Müller and Sarah Guido

### 6. **TensorFlow and PyTorch (Deep Learning Frameworks)**
   - **Core Concepts**:
     - **Tensors**: Multi-dimensional arrays for handling complex datasets.
     - **Neural Networks**: Defining and training neural network models.
     - **Autograd (Automatic Differentiation)**: Computing gradients for optimization.
     - **Backpropagation**: Updating model weights based on loss.
     - **Model Training**: Using GPUs for accelerated deep learning.
   - **Applications**: These frameworks are used for building, training, and deploying deep learning models, including neural networks for computer vision, natural language processing, reinforcement learning, and more.

   **Resources**:
   - "Deep Learning with Python" by François Chollet (for TensorFlow/Keras)
   - PyTorch official documentation and tutorials

### 7. **Keras (High-level Deep Learning API)**
   - **Core Concepts**:
     - **Sequential and Functional API**: Building neural networks layer by layer.
     - **Activation Functions**: ReLU, Sigmoid, Softmax, etc.
     - **Loss Functions**: Mean Squared Error, Cross-Entropy, etc.
     - **Optimizers**: SGD, Adam, RMSprop.
     - **Callbacks**: Early stopping, model checkpointing.
   - **Applications**: Keras (integrated with TensorFlow) allows for fast prototyping and easy experimentation with deep learning models.

   **Resources**:
   - Keras official documentation  
   - "Deep Learning with Python" by François Chollet

### 8. **Statsmodels (Statistical Modeling)**
   - **Core Concepts**:
     - **Linear Regression, ANOVA**: Advanced linear modeling and statistical analysis.
     - **Time Series Analysis**: Autoregressive models (AR), Moving Average (MA), ARIMA.
     - **Hypothesis Testing**: Conducting statistical tests like t-tests, chi-square tests.
   - **Applications**: Statsmodels is key for statistical data analysis, providing advanced tools for hypothesis testing and model fitting, especially useful in econometrics and time-series forecasting.

   **Resources**:
   - Statsmodels official documentation
   - "Python for Data Analysis" by Wes McKinney

### 9. **SciPy (Scientific Computing)**
   - **Core Concepts**:
     - **Optimization**: Non-linear optimization methods.
     - **Signal Processing**: Fourier transforms, filtering.
     - **Statistical Functions**: Descriptive stats, distributions, hypothesis testing.
     - **Linear Algebra and Integration**: Matrix operations, solving equations.
   - **Applications**: SciPy extends NumPy’s capabilities, providing efficient numerical operations for optimization, integration, interpolation, and more, which are crucial for machine learning algorithms.

   **Resources**:
   - SciPy official documentation  
   - "Numerical Python" by Robert Johansson

### 10. **Jupyter Notebooks (Interactive Development Environment)**
   - **Core Concepts**:
     - **Interactive Coding**: Writing and executing Python code in cells.
     - **Markdown Support**: Documenting code with explanations.
     - **Visualizing Data**: Displaying plots inline for quick analysis.
   - **Applications**: Jupyter notebooks are widely used for experimenting with machine learning models, data analysis, and prototyping AI workflows in an interactive manner.

   **Resources**:
   - Jupyter official documentation

### 11. **Natural Language Processing (NLP) Libraries**
   - **Core Concepts**:
     - **Text Preprocessing**: Tokenization, stemming, lemmatization, stopword removal.
     - **Word Embeddings**: Word2Vec, GloVe, BERT for semantic understanding of text.
     - **Sentiment Analysis, Topic Modeling**: Basic tasks in NLP.
   - **Libraries**:
     - **NLTK (Natural Language Toolkit)**: Basic NLP tasks like tokenization, stemming, and POS tagging.
     - **spaCy**: High-performance NLP library for named entity recognition, dependency parsing, and other advanced tasks.
   - **Applications**: NLP is crucial for AI applications like text classification, chatbot development, and language translation.

   **Resources**:
   - "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper (for NLTK)
   - spaCy official documentation

### 12. **Regular Expressions (Regex)**
   - **Core Concepts**:
     - **Pattern Matching**: Finding specific patterns in text using regex.
     - **Substitution**: Replacing or modifying text using patterns.
     - **Advanced Search**: Grouping, assertions, and special characters in regex.
   - **Applications**: Regex is vital for text preprocessing in machine learning tasks, especially in NLP and data cleaning.

   **Resources**:
   - "Automate the Boring Stuff with Python" (has a section on regex)

### 13. **Joblib and Pickle (Model Persistence)**
   - **Core Concepts**:
     - **Serialization**: Saving trained models to disk for future use.
     - **Joblib**: Optimized for storing large NumPy arrays, faster for model persistence.
     - **Pickle**: Python’s built-in tool for serializing Python objects.
   - **Applications**: After training machine learning models, saving and reloading them efficiently is crucial for deploying models in production.

   **Resources**:
   - Python’s official documentation (Pickle and Joblib)

### 14. **Pipelines (Automating Machine Learning Workflows)**
   - **Core Concepts**:
    - **Preprocessing Pipelines**: Automating data cleaning, feature engineering, and transformation.
    - **Cross-Validation Pipelines**: Combining model training and validation into a streamlined workflow.
    - **Scikit-learn Pipelines**: Tools for chaining together transformers and estimators.

