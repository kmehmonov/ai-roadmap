### Linear Regression

#### Model Definition

Define the model for a single-variable linear regression:

\[
y_i = a + b \cdot x_i + \varepsilon_i
\]

\[
\hat{y}_i = a + b \cdot x_i
\]

where:
- \( y_i \) is the actual output for the \( i \)-th observation.
- \( \hat{y}_i \) is the predicted output for the \( i \)-th observation.
- \( \varepsilon_i \) is the error term for the \( i \)-th observation.

---

#### Objective Function: Sum of Squared Errors

The goal is to minimize the sum of squared errors \( f \):

\[
f = \sum\limits_{i=1}^n \left(y_i - (a + b \cdot x_i) \right)^2
\]

or equivalently,

\[
f(a, b) = \sum\limits_{i=1}^n \left(y_i - (a + b \cdot x_i) \right)^2
\]

---

#### Minimization Condition

To minimize \( f(a, b) \), take partial derivatives with respect to \( a \) and \( b \) and set them to zero:

\[
\begin{cases}
    \frac{\partial f(a, b)}{\partial a} = 0 \\
    \frac{\partial f(a, b)}{\partial b} = 0
\end{cases}
\]

#### Applying the Partial Derivatives

1. **Partial derivative with respect to \( a \):**
   \[
   -2 \sum\limits_{i=1}^n \left(y_i - a - b \cdot x_i \right) = 0
   \]

2. **Partial derivative with respect to \( b \):**
   \[
   -2 \sum\limits_{i=1}^n x_i \left(y_i - a - b \cdot x_i \right) = 0
   \]

---

#### Setting Up the Equations

The following two equations are obtained:

\[
\begin{cases}
    \sum\limits_{i=1}^n y_i = a \cdot n + b \sum\limits_{i=1}^n x_i \\
    \sum\limits_{i=1}^n y_i x_i = a \sum\limits_{i=1}^n x_i + b \sum\limits_{i=1}^n x_i^2
\end{cases}
\]

or, using summation notation:

\[
\begin{cases}
    \sum y = a \cdot n + b \cdot \sum x \\
    \sum yx = a \cdot \sum x + b \cdot \sum x^2
\end{cases}
\]

---

#### Converting to Mean Form

Dividing through by \( n \) gives equations in terms of means:

\[
\begin{cases}
    \bar{y} = a + b \cdot \bar{x} \\
    \overline{yx} = a \cdot \bar{x} + b \cdot \overline{x^2}
\end{cases}
\]

#### Solving in Matrix Form

Express the equations in matrix notation:

\[
\begin{pmatrix} 1 & \bar{x} \\ \bar{x} & \overline{x^2} \end{pmatrix}
\begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} \bar{y} \\ \overline{yx} \end{pmatrix}
\]

To solve for \( a \) and \( b \):

\[
\begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} 1 & \bar{x} \\ \bar{x} & \overline{x^2} \end{pmatrix}^{-1} \begin{pmatrix} \bar{y} \\ \overline{yx} \end{pmatrix}
\]
