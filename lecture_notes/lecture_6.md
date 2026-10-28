# Deep Learning — Automatic Differentiation (Backprop / Autograd)

> Lecture notes based on *Deep Learning, Brian Kulis* — "Automatic Differentiation".
> Directly useful for **Homework 2**.

## Table of Contents

1. [Motivation](#1-motivation)
2. [Convention: Right Matrix-Multiplies](#2-convention-right-matrix-multiplies)
3. [Recap of Freshman Calculus](#3-recap-of-freshman-calculus)
4. [The Chain Rule](#4-the-chain-rule)
5. [Computational Graphs](#5-computational-graphs)
6. [Backpropagation (Reverse-Mode Differentiation)](#6-backpropagation-reverse-mode-differentiation)
7. [Beyond Scalars](#7-beyond-scalars)
8. [Homework Setup: Simplified Graphs](#8-homework-setup-simplified-graphs)
9. [DIY Tricks: `np.einsum`](#9-diy-tricks-npeinsum)
10. [Tensors and Tensor Contraction](#10-tensors-and-tensor-contraction)
11. [Derivatives of Tensor-Valued Functions](#11-derivatives-of-tensor-valued-functions)
12. [Summary Cheat Sheet](#12-summary-cheat-sheet)

---

## 0. Recap

**Quiz 1 (linear regression).** With data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, targets $\mathbf{y} \in \mathbb{R}^{n}$ and weights $\mathbf{w} \in \mathbb{R}^{d}$:

```math
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}
\qquad
\mathcal{L}(\mathbf{w}) = \frac{1}{n}\lVert \mathbf{y} - \mathbf{X}\mathbf{w} \rVert_2^2
```

Setting the gradient to zero gives the normal equations and the closed-form solution (assuming $\mathbf{X}^T\mathbf{X}$ is invertible, i.e. $\mathbf{X}$ has full column rank):

```math
\nabla_{\mathbf{w}}\mathcal{L} = -\frac{2}{n}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) = 0
\quad\Longrightarrow\quad
\mathbf{w}^{\ast} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
```

**Last time: ReLU MLP.** A modern MLP alternates *linear blocks* and *ReLU blocks*:

```
x → [Linear] → [ReLU] → [Linear] → [ReLU] → ... → output
```

ReLU networks are **piecewise linear** functions of the input.

---

## 1. Motivation

A classification network looks like:

```
x → [Linear W1] → [ReLU] → [Linear W2] → [ReLU] → [Linear W3] → [Softmax] → [Cross Entropy] → L(W1, W2, W3)
```

We need the gradients $\nabla_{W_i}\mathcal{L}(W_1\ W_2\ W_3)$ for every $i$.

- Coding these by hand for one network is painful.
- Every time the architecture changes, the hand-written code must change too.
- **Automatic differentiation** solves this: describe the function once as a graph, and gradients come out automatically.

---

## 2. Convention: Right Matrix-Multiplies

- In math class we usually multiply on the **left**: $W\mathbf{x}$.
- In this lecture we multiply on the **right**: $\mathbf{x}W$.
- Reason: it easily accommodates a **batch dimension** (each row of $X$ is one example).

A matrix $W \in \mathbb{R}^{n \times m}$ takes inputs in $\mathbb{R}^{n}$ (row vectors) and produces outputs in $\mathbb{R}^{m}$:

```math
\underbrace{\mathbf{x}}_{1 \times n}\ \underbrace{W}_{n \times m} = \underbrace{\mathbf{y}}_{1 \times m}
\qquad
\underbrace{X}_{B \times n}\ \underbrace{W}_{n \times m} = \underbrace{Y}_{B \times m}
```

---

## 3. Recap of Freshman Calculus

### 3.1 What is a derivative, really?

Let $f : \mathbb{R}^{n} \to \mathbb{R}^{m}$. The derivative at $\mathbf{x}$ is a **linear map** (a matrix):

```math
D[f](\mathbf{x}) \in \mathbb{R}^{n \times m}
```

Its input and output dimensions match those of $f$ (under the right-multiply convention).

### 3.2 Derivative formula

```math
D[f](\mathbf{x})_{ij} = \frac{\partial f_j}{\partial x_i}(\mathbf{x})
```

- Row index $i$ = input coordinate, column index $j$ = output coordinate.
- $f_j$ is the $j$-th coordinate of $f$.
- Note: this is the **transpose** of the usual Jacobian, because we right-multiply.

### 3.3 Special case: gradients

For a scalar function $f(x_1\ x_2\ x_3) = x_1 x_2 + x_3$:

```math
\nabla f(x_1\ x_2\ x_3) = \begin{bmatrix} x_2 \\ x_1 \\ 1 \end{bmatrix}
```

This is a $3 \times 1$ matrix: $f$ takes 3-dimensional inputs and gives 1-dimensional outputs.

### 3.4 Example 1

$f(\mathbf{x}) = (x_1^2\ \ x_1 x_2 x_3)$, mapping $\mathbb{R}^3 \to \mathbb{R}^2$. The derivative is $3 \times 2$:

```math
D[f](\mathbf{x}) =
\begin{bmatrix}
2x_1 & x_2 x_3 \\
0 & x_1 x_3 \\
0 & x_1 x_2
\end{bmatrix}
```

### 3.5 Example 2

$f(x_1\ x_2) = (x_2^2\ \ x_1 + x_2\ \ x_1 x_2)$, mapping $\mathbb{R}^2 \to \mathbb{R}^3$:

- $f_1 = x_2^2$
- $f_2 = x_1 + x_2$
- $f_3 = x_1 x_2$

The derivative is $2 \times 3$:

```math
D[f](\mathbf{x}) =
\begin{bmatrix}
0 & 1 & x_2 \\
2x_2 & 1 & x_1
\end{bmatrix}
```

---

## 4. The Chain Rule

### 4.1 Scalar vs. matrix form

Scalar (freshman calculus):

```math
(f \circ g)'(x) = f'(g(x))\ g'(x)
```

Higher dimensions, with $g : \mathbb{R}^{n} \to \mathbb{R}^{m}$ and $f : \mathbb{R}^{m} \to \mathbb{R}^{k}$:

```
R^n ──g──▶ R^m ──f──▶ R^k
```

```math
D[f \circ g](\mathbf{x}) = D[g](\mathbf{x})\ D[f](g(\mathbf{x}))
```

**Dimension check:**

```math
\underbrace{D[g](\mathbf{x})}_{n \times m}\ \underbrace{D[f](g(\mathbf{x}))}_{m \times k} = \underbrace{D[f \circ g](\mathbf{x})}_{n \times k}
```

> With right-multiplication, the derivatives appear in the **same order as the data flows** (first $g$, then $f$).

### 4.2 Example

$g(x) = x^4$ and $f(y) = \sqrt{y}$, so $f(g(x)) = x^2$.

```math
D[g](x) = 4x^3
\qquad
D[f](y) = \frac{1}{2\sqrt{y}}
```

```math
D[f \circ g](x) = D[g](x)\ D[f](g(x)) = 4x^3 \cdot \frac{1}{2x^2} = 2x \ \checkmark
```

> 🕰️ **Intuition — a watch gear train.** In a mechanical movement, the barrel drives the center wheel, which drives the third wheel, which drives the fourth (seconds) wheel. If the center wheel turns $r_1$ times per barrel turn and the third wheel turns $r_2$ times per center-wheel turn, then the seconds wheel's speed relative to the barrel is the **product** $r_1 \cdot r_2 \cdot r_3$. Each wheel only "knows" its own ratio with its neighbor — exactly like each block in the chain rule only knows its own local derivative. Backprop is just multiplying the gear ratios from the seconds wheel back to the barrel.

### 4.3 Multiple inputs

Let $g(\mathbf{x}\ \mathbf{y}) : \mathbb{R}^{n} \times \mathbb{R}^{m} \to \mathbb{R}^{k}$. The partial derivatives are defined analogously:

```math
D_{\mathbf{x}}[g](\mathbf{x}\ \mathbf{y}) \in \mathbb{R}^{n \times k}
\qquad
D_{\mathbf{x}}[g](\mathbf{x}\ \mathbf{y})_{ij} = \frac{\partial g_j}{\partial x_i}(\mathbf{x}\ \mathbf{y})
```

### 4.4 Chain rule with multiple inputs

With $f : \mathbb{R}^{k} \to \mathbb{R}^{l}$, the chain rule works the same way:

```math
D_{\mathbf{x}}[f \circ g](\mathbf{x}\ \mathbf{y}) = D_{\mathbf{x}}[g](\mathbf{x}\ \mathbf{y})\ D[f](g(\mathbf{x}\ \mathbf{y}))
```

### 4.5 Re-using inputs (the input "splits")

Let $g(\mathbf{x}\ \mathbf{y}) : \mathbb{R}^{n} \times \mathbb{R}^{n} \to \mathbb{R}^{m}$ and define $f(\mathbf{x}) = g(\mathbf{x}\ \ 2\mathbf{x})$.

```
        ┌──────────────────▶ (first input of g)
x ──dup─┤                                      g ──▶ R^m
        └──── ×2 ──────────▶ (second input of g)
```

```math
D[f](\mathbf{x}) = D_{\mathbf{x}}[g](\mathbf{x}\ \ 2\mathbf{x}) + 2\ D_{\mathbf{y}}[g](\mathbf{x}\ \ 2\mathbf{x})
```

> **Rule:** when a variable is used in several places (the paths split), **sum** the derivatives from each path.

### 4.6 More complicated compositions

Let

- $f : \mathbb{R}^{k} \to \mathbb{R}^{p}$
- $g(\mathbf{x}\ \mathbf{y}) : \mathbb{R}^{n} \times \mathbb{R}^{m} \to \mathbb{R}^{k}$
- $h(\mathbf{z}\ \mathbf{w}) : \mathbb{R}^{l} \times \mathbb{R}^{q} \to \mathbb{R}^{m}$

and consider $F(\mathbf{x}\ \mathbf{z}\ \mathbf{w}) = f(g(\mathbf{x}\ \ h(\mathbf{z}\ \mathbf{w})))$.

```
z ─┐
   ├─ h ──▶ y ─┐
w ─┘           ├─ g ──▶ R^k ── f ──▶ R^p
          x ───┘
```

Following the path from $\mathbf{z}$ to the output:

```math
D_{\mathbf{z}}[F] = D_{\mathbf{z}}[h](\mathbf{z}\ \mathbf{w})\ D_{\mathbf{y}}[g](\mathbf{x}\ \ h(\mathbf{z}\ \mathbf{w}))\ D[f](g(\mathbf{x}\ \ h(\mathbf{z}\ \mathbf{w})))
```

### 4.7 Worked example (forward then backward)

$f(x) = x^2 \qquad g(x\ y) = xy \qquad h(z\ w) = z + w$

so $F(x\ z\ w) = x^2 (z+w)^2$. Evaluate at $x = 2,\ z = 2,\ w = 3$.

**Direct:**

```math
D_z[F] = \frac{\partial}{\partial z}\left[x^2 (z+w)^2\right] = 2x^2(z+w) = 2 \cdot 4 \cdot 5 = 40
```

**Step-by-step (how autodiff does it):**

1. **Plug in variables:** $x = 2,\ z = 2,\ w = 3$.
2. **Forward pass:** $y = h(z\ w) = 5$, then $g = xy = 10$, then $f = 100$.
3. **Compute and store each block's local derivative:**
   - $D_z[h] = 1$
   - $D_y[g] = x = 2$
   - $D[f](10) = 2 \cdot 10 = 20$
4. **Multiply along the path:**

```math
D_z[F] = 1 \cdot 2 \cdot 20 = 40 \ \checkmark
```

---

## 5. Computational Graphs

A **computational graph** is a **directed acyclic graph (DAG)** with two kinds of nodes:

| Node type | Represents | Holds |
|---|---|---|
| **Variable node** | function inputs and/or outputs | the value (real number / vector / matrix) |
| **Operation node** | a function | any state needed for forward/backward (e.g. stored derivatives) |

Properties:

- The graph is **bipartite**: variable nodes connect only to operation nodes, and vice versa.
- All **source** and **sink** nodes are **variable** nodes.

```
(x) ──▶ [ g ] ──▶ (y) ──▶ [ f ] ──▶ (z)
 var      op      var       op      var
```

---

## 6. Backpropagation (Reverse-Mode Differentiation)

**Backpropagation** automatically computes the gradient of a function specified by a computational graph. It has two phases:

1. **Forward pass** — source ➜ sink: compute values (and store local derivatives).
2. **Backward pass** — sink ➜ source: compute the gradient of the sink with respect to every variable.

> ⚠️ You **must** do a forward pass before the backward pass.

### 6.1 Forward pass algorithm

1. Seed the source variable node with a value, e.g. $x = 3$.
2. Find all operation nodes whose inputs all have values assigned.
3. Compute those operations, assign outputs to their output variable nodes, and (optionally) store state such as the local derivative.
4. Repeat until the sink has a value.

### 6.2 Example: a simple chain

$y = g(x) = x^4 \qquad z = f(y) = \sqrt{y}$

**Forward (with stored derivatives):**

| Step | Value | Stored in op node |
|---|---|---|
| seed | $x = 3$ | |
| $g$ | $y = 81$ | $D_x[y] = 4x^3 = 108$ |
| $f$ | $z = 9$ | $D_y[z] = \frac{1}{2\sqrt{y}} = \frac{1}{18}$ |

**Backward:**

```math
D_y[z] = \frac{1}{18}
\qquad
D_x[z] = D_x[y]\ D_y[z] = \frac{108}{18} = 6
```

Check: $z = x^2$, so $\frac{dz}{dx} = 2x = 6 \ \checkmark$

### 6.3 Example: a branching graph

$c = h(a\ b) = a + b \qquad e = g(c\ d) = cd \qquad L = f(e) = e^2$

```
a ─┐
   ├─ h(+) ──▶ c ─┐
b ─┘              ├─ g(×) ──▶ e ── f(²) ──▶ L
             d ───┘
```

**Forward** with $a = 3,\ b = 2,\ d = 1$:

| Op | Output | Stored local derivatives |
|---|---|---|
| $h$ | $c = 5$ | $D_a[c] = 1\quad D_b[c] = 1$ |
| $g$ | $e = 5$ | $D_c[e] = d = 1\quad D_d[e] = c = 5$ |
| $f$ | $L = 25$ | $D_e[L] = 2e = 10$ |

**Backward** (each op multiplies the incoming gradient by its stored derivatives):

```math
D_e[L] = 10
```

```math
D_c[L] = D_c[e]\ D_e[L] = 1 \cdot 10 = 10
\qquad
D_d[L] = D_d[e]\ D_e[L] = 5 \cdot 10 = 50
```

```math
D_a[L] = D_a[c]\ D_c[L] = 10
\qquad
D_b[L] = D_b[c]\ D_c[L] = 10
```

### 6.4 Backprop is *local*

Each operation node only needs to see its **immediate neighbors**:

- **Forward:** an op node needs only its input values to compute its output and its local derivatives.
- **Backward:** an op node needs only the gradient w.r.t. its output plus its stored local derivatives to produce gradients w.r.t. its inputs.

### 6.5 How do you code it up?

Operation nodes hold pointers to their input and output variable nodes.

```python
class Operation:
    def forward(self, x):
        # 1. Compute the output from the input and assign it to the output node(s).
        # 2. Compute and store relevant state (e.g. the local derivative).
        ...

    def backward(self, grad_output):
        # 1. Input = derivative of the final output w.r.t. this op's output.
        # 2. Multiply by the stored derivatives to get derivatives
        #    w.r.t. this op's inputs, and pass them back.
        ...
```

---

## 7. Beyond Scalars

### 7.1 Vector example

$\mathbf{y} = (x_1^2\ \ x_1 x_2)$ and $z = y_1 y_2$, with $\mathbf{x} = (2\ \ 3)$.

**Forward:**

- $\mathbf{y} = (4\ \ 6)$ and

```math
D_{\mathbf{x}}[\mathbf{y}] =
\begin{bmatrix}
2x_1 & x_2 \\
0 & x_1
\end{bmatrix}
=
\begin{bmatrix}
4 & 3 \\
0 & 2
\end{bmatrix}
```

- $z = 24$ and

```math
D_{\mathbf{y}}[z] = \begin{bmatrix} y_2 \\ y_1 \end{bmatrix} = \begin{bmatrix} 6 \\ 4 \end{bmatrix}
```

**Backward:**

```math
D_{\mathbf{x}}[z] = D_{\mathbf{x}}[\mathbf{y}]\ D_{\mathbf{y}}[z] =
\begin{bmatrix}
4 & 3 \\
0 & 2
\end{bmatrix}
\begin{bmatrix} 6 \\ 4 \end{bmatrix}
=
\begin{bmatrix} 36 \\ 8 \end{bmatrix}
```

**Check:** $z = x_1^3 x_2$, so

```math
D_{\mathbf{x}}[z] = \begin{bmatrix} 3x_1^2 x_2 \\ x_1^3 \end{bmatrix} = \begin{bmatrix} 36 \\ 8 \end{bmatrix} \ \checkmark
```

### 7.2 Even simple functions are operations

Something like $(xy + z)^2$ is itself a graph of elementary operations:

```
x ─┐
   ├─ [×] ──▶ (xy) ─┐
y ─┘                ├─ [+] ──▶ (xy+z) ── [^2] ──▶ output
               z ───┘
```

### 7.3 Derivatives of batched functions

Given $f : \mathbb{R}^{n} \to \mathbb{R}^{m}$, its **batched** version applies $f$ to each row:

```math
f_{\text{batched}} : \mathbb{R}^{B \times n} \to \mathbb{R}^{B \times m}
\qquad
f_{\text{batched}}(X)[i] = f(X[i])
```

- All functions on the homework are batched functions.
- To differentiate a batched function, **compute the derivative of each row individually**.

**Example** (same $f$ as 7.1, batch of 2):

```math
X = \begin{bmatrix} 2 & 3 \\ 1 & 5 \end{bmatrix}
\ \longrightarrow\
Y = \begin{bmatrix} 4 & 6 \\ 1 & 5 \end{bmatrix}
\ \longrightarrow\
\mathbf{z} = \begin{bmatrix} 24 \\ 5 \end{bmatrix}
```

Per-row derivatives:

```math
D_{\mathbf{x}}[\mathbf{y}] :\ 
\begin{bmatrix} 4 & 3 \\ 0 & 2 \end{bmatrix}
\quad
\begin{bmatrix} 2 & 5 \\ 0 & 1 \end{bmatrix}
\qquad\qquad
D_{\mathbf{y}}[z] :\ 
\begin{bmatrix} 6 \\ 4 \end{bmatrix}
\quad
\begin{bmatrix} 5 \\ 1 \end{bmatrix}
```

### 7.4 Derivatives of matrix functions

For $f : \mathbb{R}^{n \times m} \to \mathbb{R}^{k}$:

1. **Unroll** (flatten) the matrix input into a vector in $\mathbb{R}^{nm}$.
2. Differentiate as usual.
3. **Reshape** the gradient back into the shape of the matrix.

---

## 8. Homework Setup: Simplified Graphs

On the homework, only consider **chain-like graphs of layers**:

```
x ──▶ [ Layer ] ──▶ [ Layer ] ──▶ [ Layer ] ──▶ L
         w₁            w₂            w₃
```

- You only need gradients for the variables **inside** layers (the parameters $\mathbf{w}$).
- Only consider the case $L \in \mathbb{R}$ (scalar loss), so $D_{\mathbf{v}}[L]$ always has the **same shape** as $\mathbf{v}$.

### The backward function of a layer

A layer with input $\mathbf{x}$, parameter $\mathbf{w}$ and output $\mathbf{y}$ receives $D_{\mathbf{y}}[L]$ from the layer after it and:

1. Computes $D_{\mathbf{x}}[L]$ and **passes it back** to the previous layer.
2. Computes $D_{\mathbf{w}}[L]$ and **stores** it (for the optimizer update).

```
          D_x[L] ◀──  [ Layer (w) ]  ◀── D_y[L]
                         │
                    stores D_w[L]
```

---

## 9. DIY Tricks: `np.einsum`

A super-useful function for expressing complicated (especially **batched**) tensor manipulations.
Tutorial: <https://ajcr.net/Basic-guide-to-einsum/>

**Rule:** each input gets an index string; indices **not** in the output are **summed over**.

### 9.1 Example 1

```python
Z = np.einsum('abc,bad->ab', X, Y)
```

- `X` has shape `[A, B, C]`, `Y` has shape `[B, A, D]`, output `Z` has shape `[A, B]`.

```math
Z[a\ b] = \sum_{c}\sum_{d} X[a\ b\ c] \cdot Y[b\ a\ d]
```

### 9.2 Example 2

```python
W = np.einsum('abcd,daa,be->ade', X, Y, Z)
```

- Inputs have shapes `[A, B, C, D]`, `[D, A, A]`, `[B, E]`; output has shape `[A, D, E]`.

```math
W[a\ d\ e] = \sum_{b}\sum_{c} X[a\ b\ c\ d] \cdot Y[d\ a\ a] \cdot Z[b\ e]
```

> Note: the slide writes the output as `adf`; since `f` appears in no input, the intended output index is presumably `e`.

### 9.3 Familiar operations

| einsum | Formula | Meaning |
|---|---|---|
| `np.einsum('ij,jk->ik', X, Y)` | $Z[i\ k] = \sum_j X[i\ j] \cdot Y[j\ k]$ | matrix multiplication |
| `np.einsum('ij->ji', X)` | $Z[j\ i] = X[i\ j]$ | transpose |
| `np.einsum('ii->', X)` | $Z = \sum_i X[i\ i]$ | trace |

### 9.4 Batched matrix multiplication

- $X \in \mathbb{R}^{B \times n}$: a batch of $n$-dimensional vectors.
- $M \in \mathbb{R}^{B \times n \times m}$: a batch of $n \times m$ matrices.
- Want $Y \in \mathbb{R}^{B \times m}$ whose $i$-th row is $X[i]\ M[i] \in \mathbb{R}^{m}$.

```math
Y[i\ j] = \sum_{k} X[i\ k] \cdot M[i\ k\ j]
```

```python
Y = np.einsum('ik,ikj->ij', X, M)
```

---

## 10. Tensors and Tensor Contraction

### 10.1 Definitions

- A **matrix** in $\mathbb{R}^{n \times m}$ is an $n$ by $m$ 2-D array — a "list of lists" (outer size $n$, inner size $m$).
- A **tensor** in $\mathbb{R}^{n_1 \times \cdots \times n_k}$ is a $k$-dimensional array — a "list of lists of … of lists".
- A tensor in $\mathbb{R}^{n \times m \times p}$ has **shape** `[n, m, p]`.
- Access via multi-indices, e.g. `T[0, 2, 3]`.

### 10.2 Contraction along one dimension

Tensor contraction **generalizes matrix multiplication**. Tensors $A$ and $B$ can be contracted along one dimension if the **last** entry of $A$'s shape equals the **first** entry of $B$'s shape.

- $A$: shape `[2, 45, 7]`, $B$: shape `[7, 3, 67]`
- Output shape: drop the shared entry and concatenate ➜ `contract(A, B)` has shape `[2, 45, 3, 67]`.

If $A$ has shape $[n_1\ \ldots\ n_k\ \ p]$ and $B$ has shape $[p\ \ m_1\ \ldots\ m_l]$:

```math
\text{contract}(A\ B)[i_1\ \ldots\ i_k\ \ j_1\ \ldots\ j_l]
= \sum_{x=1}^{p} A[i_1\ \ldots\ i_k\ \ x] \cdot B[x\ \ j_1\ \ldots\ j_l]
```

When $k = l = 1$ this is exactly matrix multiplication:

```math
\text{contract}(A\ B)[i\ j] = \sum_{x=1}^{p} A[i\ x] \cdot B[x\ j]
```

### 10.3 Contraction along many dimensions

Contraction along $d$ dimensions requires the **last $d$** dimensions of $A$ to match the **first $d$** dimensions of $B$.

- $\text{shape}(A) = [2\ 5\ 4\ 6]$, $\text{shape}(B) = [4\ 6\ 8\ 2]$ ➜ contractible along 2 dimensions.
- `contract(A, B, 2)` has shape `[2, 5, 8, 2]`.
- In NumPy this is `np.tensordot(A, B, axes=2)`.

General formula — if $A$ has shape $[n_1\ \ldots\ n_k\ \ p_1\ \ldots\ p_d]$ and $B$ has shape $[p_1\ \ldots\ p_d\ \ m_1\ \ldots\ m_l]$:

```math
\text{contract}(A\ B\ d)[i_1\ \ldots\ i_k\ \ j_1\ \ldots\ j_l]
= \sum_{x_1=1}^{p_1} \cdots \sum_{x_d=1}^{p_d}
A[i_1\ \ldots\ i_k\ \ x_1\ \ldots\ x_d] \cdot B[x_1\ \ldots\ x_d\ \ j_1\ \ldots\ j_l]
```

### 10.4 Why contraction?

Contraction lets us write **linear maps between arbitrary tensor shapes** as tensors of larger shape.

If $T$ has shape `[2, 3, 4, 5]`, it defines a linear map from $2 \times 3$ matrices to $4 \times 5$ matrices:

```math
L(X) = \text{contract}(X\ T\ 2)
```

---

## 11. Derivatives of Tensor-Valued Functions

### 11.1 General case

Let $f : \mathbb{R}^{n_1 \times n_2 \times n_3} \to \mathbb{R}^{m_1 \times m_2}$. Then

```math
D[f](X) \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times m_1 \times m_2}
```

It is a tensor that contracts **all** input dimensions and produces the output dimensions, i.e. a linear map $\mathbb{R}^{n_1 \times n_2 \times n_3} \to \mathbb{R}^{m_1 \times m_2}$:

```math
D[f](X)[i\ j\ k\ a\ b] = \frac{\partial f_{a b}}{\partial X_{i j k}}(X)
```

### 11.2 Scalar-valued functions (the loss!)

Let $f : \mathbb{R}^{n_1 \times n_2 \times n_3} \to \mathbb{R}$. Then $D[f](X) \in \mathbb{R}^{n_1 \times n_2 \times n_3 \times 1}$, and we usually **drop the $\times 1$**, so $D[f](X)$ has the **same shape as $X$**.

The contraction becomes an elementwise product followed by a sum:

```math
\text{contract}(V\ \ D[f](X)) = \text{sum}\left(V \odot D[f](X)\right)
```

- $\odot$ is the coordinate-wise product (normal `*` in NumPy).
- `sum` adds all entries.

---

## 12. Summary Cheat Sheet

| Concept | Formula / Rule |
|---|---|
| Derivative (right-multiply) | $D[f](\mathbf{x}) \in \mathbb{R}^{n \times m}$ with entries $\frac{\partial f_j}{\partial x_i}$ |
| Chain rule | $D[f \circ g](\mathbf{x}) = D[g](\mathbf{x})\ D[f](g(\mathbf{x}))$ |
| Re-used input | **sum** the derivatives along every path |
| Forward pass | source ➜ sink, compute values and store local derivatives |
| Backward pass | sink ➜ source, multiply incoming gradient by stored derivatives |
| Locality | each op node only needs its immediate neighbors |
| Batched functions | differentiate each row independently |
| Matrix inputs | flatten, differentiate, reshape back |
| Scalar loss | $D_{\mathbf{v}}[L]$ has the same shape as $\mathbf{v}$ |
| Layer backward | given $D_{\mathbf{y}}[L]$: return $D_{\mathbf{x}}[L]$, store $D_{\mathbf{w}}[L]$ |
| einsum | indices missing from output are summed |
| Tensor contraction | `np.tensordot(A, B, axes=d)` |
