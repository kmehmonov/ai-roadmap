## Solving Linear Regression

$$
y_i = a + b \cdot x_i + \varepsilon_i
$$

$$
\hat{y}_i = a + b \cdot x_i
$$

$y_i \to$ actual output for $i$th observation.
$\hat{y}_i \to$ predicted output for $i$th observation.
$\varepsilon_i \to$ error term for $i$th observation.

---

$$
f = \varepsilon_1^2 + \varepsilon_2^2 + \dots + \varepsilon_i^2
$$

$$
f = \sum\limits_{i=1}^n \left(y_i - (a+b\cdot x_i) \right)^2
$$

$$
f(a, b) = \sum\limits_{i=1}^n \left[y_i - (a+b\cdot x_i) \right]^2
$$

---

$f(a,b) \to min$

$$
\begin{cases}
    \frac{df(a, b)}{da} = 0 \\
    \frac{f(a, b)}{db} = 0
\end{cases}
$$

$$
\begin{cases}
    -2\sum\limits_{i=1}^n \left(y_i - a - b\cdot x_i \right) = 0 \\
    -2 x_i \sum\limits_{i=1}^n \left( y_i - a - b \cdot x_i \right) = 0
\end{cases}
$$

$$
\begin{cases}
    \sum\limits_{i=1}^n y_i - \sum\limits_{i=1}^n a - \sum\limits_{i=1}^n x_i^2 = 0 \\
    \sum\limits_{i=1}^n y_i x_i - \sum\limits_{x=1}^n a x_i - \sum\limits_{i=1}^n b x_i^2 = 0
\end{cases}
$$

$$
\begin{cases}
    \sum y - a \cdot n - b \cdot \sum = 0 \\
    \sum yx - a\sum x - b\sum x^2 = 0
\end{cases}
$$

$$
\begin{cases}
    \bar{y} - a - b \bar{x} = 0 \\
    \bar{yx} - a \bar{x} - b\bar{x^2} = 0
\end{cases}
$$

$$
\begin{cases}
    a + b\bar{x} = \bar{y} \\
    a\bar{x} + b\bar{x^2} = \bar{yx}
\end{cases}
$$

$$
\begin{pmatrix}1 & \bar{x} \\\bar{x} & \bar{x^2}\end{pmatrix}
\times
\begin{pmatrix}a \\ b\end{pmatrix} =
\begin{pmatrix}\bar{y} \\ \bar{yx} \end{pmatrix}
$$

$$
\begin{pmatrix}a \\ b\end{pmatrix} =
\begin{pmatrix} 1 & \bar{x} \\ \bar{x} & \bar{x^2} \end{pmatrix} ^ {-1}
\times
\begin{pmatrix} \bar{y} \\ \bar{yx} \end{pmatrix}
$$

---

## Rule for Matrix Multiplication

To multiply two matrices \( A \) and \( B \), the number of **columns** in \( A \) must be equal to the number of **rows** in \( B \). If:

- Matrix \( A \) has dimensions \( m \times n \) (with \( m \) rows and \( n \) columns).
- Matrix \( B \) has dimensions \( n \times p \) (with \( n \) rows and \( p \) columns).

Then the resulting matrix \( C = A \cdot B \) will have dimensions \( m \times p \).

### Matrix Multiplication Process

For each element in the resulting matrix \( C \), we calculate it as follows:

\[
C[i, j] = \sum\_{k=1}^{n} A[i, k] \cdot B[k, j]
\]

This means each element \( C[i, j] \) is the dot product of the \( i \)-th row of \( A \) with the \( j \)-th column of \( B \).

### Example 1: Simple Matrix Multiplication

Suppose we have two matrices:

\[
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad \text{and} \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
\]

Both matrices are \( 2 \times 2 \), so they can be multiplied. The result \( C = A \cdot B \) will also be a \( 2 \times 2 \) matrix.

Let's compute each element in \( C \):

1. **Calculate \( C[1, 1] \):**  
   \[
   C[1, 1] = (1 \cdot 5) + (2 \cdot 7) = 5 + 14 = 19
   \]

2. **Calculate \( C[1, 2] \):**  
   \[
   C[1, 2] = (1 \cdot 6) + (2 \cdot 8) = 6 + 16 = 22
   \]

3. **Calculate \( C[2, 1] \):**  
   \[
   C[2, 1] = (3 \cdot 5) + (4 \cdot 7) = 15 + 28 = 43
   \]

4. **Calculate \( C[2, 2] \):**  
   \[
   C[2, 2] = (3 \cdot 6) + (4 \cdot 8) = 18 + 32 = 50
   \]

So, the resulting matrix \( C \) is:

\[
C = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
\]

### Example 2: Multiplying Matrices with Different Dimensions

Suppose \( A \) is a \( 2 \times 3 \) matrix, and \( B \) is a \( 3 \times 2 \) matrix:

\[
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{and} \quad B = \begin{bmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{bmatrix}
\]

Here, \( A \) has 3 columns, and \( B \) has 3 rows, so we can multiply them. The result \( C = A \cdot B \) will have dimensions \( 2 \times 2 \).

Let's calculate each element of \( C \):

1. **Calculate \( C[1, 1] \):**
   \[
   C[1, 1] = (1 \cdot 7) + (2 \cdot 9) + (3 \cdot 11) = 7 + 18 + 33 = 58
   \]

2. **Calculate \( C[1, 2] \):**
   \[
   C[1, 2] = (1 \cdot 8) + (2 \cdot 10) + (3 \cdot 12) = 8 + 20 + 36 = 64
   \]

3. **Calculate \( C[2, 1] \):**
   \[
   C[2, 1] = (4 \cdot 7) + (5 \cdot 9) + (6 \cdot 11) = 28 + 45 + 66 = 139
   \]

4. **Calculate \( C[2, 2] \):**
   \[
   C[2, 2] = (4 \cdot 8) + (5 \cdot 10) + (6 \cdot 12) = 32 + 50 + 72 = 154
   \]

The resulting matrix \( C \) is:

\[
C = \begin{bmatrix} 58 & 64 \\ 139 & 154 \end{bmatrix}
\]

### Summary of Rules

1. Matrices can only be multiplied if the number of columns in the first matrix equals the number of rows in the second matrix.
2. The resulting matrix will have dimensions that combine the rows of the first matrix with the columns of the second matrix. If \( A \) is \( m \times n \) and \( B \) is \( n \times p \), then \( C = A \cdot B \) will be \( m \times p \).
3. Each element \( C[i, j] \) is calculated by taking the dot product of the \( i \)-th row of \( A \) with the \( j \)-th column of \( B \).
