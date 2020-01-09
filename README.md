# Distributed-Optimizaton
A project on Distributed optimization and ADMM for the course of Signal Processing for Big Data

## Sparce Logistic Regression (ADMM)

<br>

$minimize \; (1/2)||\sigma(A x) - b||_2^{2} + \lambda ||x||_1$

where $\sigma$ is the sigmoid fuction:

$\sigma(z) = \frac{exp(z)}{1+exp(z)} = \frac{1}{1+exp(-z)}$

<br>

**Centralized Case**:


<br>

$x^{k+1} = argmin  (l||A x - b||_2^{2} +(\rho/2)||x - z^k + u^{k}||_2^{2} )$

$z^{k+1} = S_{λ/ρ}(x^{k+1} + u^k)$

$u^{k+1} = u^{k} + x^{k+1} - z^{k+1}$

<br>

where $l$ is the *logistic regression* cost fuction or more known as *cross-entropy*:

$ l = (-1/N)\sum{b \log(\sigma(A x)) + (1-b) \log(1 - \sigma(A x))}$

or in a more compact form:

$ l = (1/N) \sum{\log(1+exp(-b x^{T}A))} $



<br>


**Consensus ( not scaled) form of the model**:

<br>

$ minimize \sum{l_i(A_i x_i - b_i) + r(z)}$, subject to $x_i − z = 0, i = 1, . . . , N$
 
where $l_i$ refers to the loss function for the ith block of
data.

<br>

Consensus ADMM algorithm for Sparse Logistic Regression problem:

$x_i^{k+1} = argmin  ( ||l_i(A_i x_i) - b_i||_2^{2} +(\rho/2)||x_i - z^k + u_i^{k}||_2^{2} )$

$z^{k+1} = S_{λ/ρN}(\overline{x}^{k+1} + \overline{u}^k)$

$u_i^{k+1} = u_i^{k} + x_i^{k+1} - z^{k+1}$

<br>

This is identical to distributed lasso, except for $x_i$ update, which here involves an $l_2$ regularized
logistic regression problem.

<br>

*Of course we can see that for N=1 the parallel and centralized cases coincide.*
