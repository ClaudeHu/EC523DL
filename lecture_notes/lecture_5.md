# Deep Learning — Neural Network Basics II

> Lecture notes (EC523, based on slides by Brian Kulis)

## Table of Contents

1. [Universal Approximation (multi-dimensional)](#1-universal-approximation-multi-dimensional)
2. [Universal Approximation in Practice](#2-universal-approximation-in-practice)
3. [Multi-Layer Perceptron (MLP)](#3-multi-layer-perceptron-mlp)
4. [Activation Functions](#4-activation-functions)
5. [Loss Functions for Classification](#5-loss-functions-for-classification)
6. [Softmax and Its Numerical Stability](#6-softmax-and-its-numerical-stability)
7. [Cross-Entropy and KL-Divergence](#7-cross-entropy-and-kl-divergence)
8. [Summary: The Standard Architecture](#8-summary-the-standard-architecture)
9. [Short Intro to PyTorch](#9-short-intro-to-pytorch)
10. [Next Time](#10-next-time)

---

## 1. Universal Approximation (multi-dimensional)

### Recap: block diagram of a single-layer network

```
x ──► [ linear block: M1, b ] ──► [ sigmoid ] ──► [ linear block: M2 ] ──► y
```

The output is

```math
y = M_2 \, \sigma(M_1 x + b)
```

where the activation $\sigma$ is applied **per coordinate**.

### The theorem

Given any (continuous) function $f : [-1, 1]^d \to \mathbb{R}$, any $\epsilon > 0$, and any **non-polynomial** activation $\sigma$, there is a sufficiently large hidden size $h$ and a single-layer network $W_2 \, \sigma(W_1 x + b)$ such that for all $x \in [-1, 1]^d$:

```math
\left| f(x) - W_2 \, \sigma(W_1 x + b) \right| \le \epsilon
```

> **Caveat:** in practice $h$ may need to be absurdly large for this theorem to be useful.

### Step 1 — the 1D case $f : [-1, 1] \to \mathbb{R}$

Build a **bump function** from the difference of two steep sigmoids:

```math
\hat{B}_\Delta(x) = \sigma\left(\frac{x + \Delta}{\Delta^2}\right) - \sigma\left(\frac{x - \Delta}{\Delta^2}\right)
```

- This is $\approx 1$ on $[-\Delta + O(\Delta^2), \; \Delta - O(\Delta^2)]$ and $\approx 0$ outside $[-\Delta, \Delta]$.
- As $\Delta \to 0$, the bump becomes sharper and narrower.
- Sum many scaled and shifted bumps to approximate any 1D function (like a histogram).

### Step 2 — the $d$-dimensional case $f : [-1, 1]^d \to \mathbb{R}$

Plan:

1. Show we can approximate $\sin(\omega \cdot x)$ and $\cos(\omega \cdot x)$ for any $\omega$.
2. Show any function can be written as a combination of $\sin(\omega \cdot x)$ and $\cos(\omega \cdot x)$ (Fourier).

The trig functions act like "bump" functions in high dimensions. Why not use real $d$-dimensional bumps? They are much harder to build than 1D bumps.

#### Key trick: $\sin(\omega \cdot x)$ is secretly a 1D function

The function only varies along the direction $\omega$ (it is a "ridge" or "corrugated sheet"). Define

```math
z := \frac{\omega}{\lVert \omega \rVert_2} \cdot x, \qquad s(z) := \sin\left(\lVert \omega \rVert_2 \, z\right), \qquad \sin(\omega \cdot x) = s(z)
```

Then $s$ is a function of one variable only.

#### Approximating $\sin(\omega \cdot x)$

1. By Step 1, approximate the 1D function $s(z) = \sin(\lVert \omega \rVert_2 \, z)$ with a single-layer net:

```math
s(z) \approx M_2 \, \sigma(M_1 z + b)
```

2. Substitute $z = \dfrac{\omega^\top x}{\lVert \omega \rVert_2}$:

```math
\sin(\omega \cdot x) \approx M_2 \, \sigma\left(\frac{M_1 \, \omega^\top}{\lVert \omega \rVert_2} x + b\right)
```

> Strictly, for $x \in [-1, 1]^d$ we have $|z| \le \sqrt{d}$, so the 1D approximation is taken on $[-\sqrt{d}, \sqrt{d}]$. This is still a bounded interval, so Step 1 applies unchanged.

This is still a single-layer network: the first weight matrix is just $M_1 \omega^\top / \lVert \omega \rVert_2$.

#### Fourier expansion

Write $f$ as

```math
f(x) = \int \left( s_\omega \sin(\omega \cdot x) + c_\omega \cos(\omega \cdot x) \right) d\omega
```

and approximate the integral with a **Riemann sum**:

```math
f(x) \approx \sum_{\omega} \delta \left( s_\omega \sin(\omega \cdot x) + c_\omega \cos(\omega \cdot x) \right)
```

More terms give a better approximation; include enough to be within $\epsilon$.

#### Putting it together

1. Each $\sin(\omega \cdot x)$ and $\cos(\omega \cdot x)$ can be approximated to within $O(\epsilon)$ by a single-layer net.
2. $f$ can be approximated to within $O(\epsilon)$ by a linear combination of those sines and cosines.
3. A linear combination of single-layer nets is itself a single-layer net (stack the hidden units side by side), so $f$ is approximated to within $O(\epsilon)$ by one single-layer net. $\blacksquare$

---

## 2. Universal Approximation in Practice

> **Being able to express a function does not mean training will find it.**

Experiment: fit a 1D target function on $[-100, 100]$ with a sigmoid single-layer net.

```python
import torch
from torch import nn

class SingleLayerNet(nn.Module):
    def __init__(self, layer_size):
        super(SingleLayerNet, self).__init__()
        self.Linear1 = nn.Linear(1, layer_size)
        self.activation = nn.Sigmoid()
        self.Linear2 = nn.Linear(layer_size, 1)

    def forward(self, x):
        out = self.Linear1(x)
        out = self.activation(out)
        out = self.Linear2(out)
        return out
```

Training loop:

```python
def train(net, num_steps=10000):
    loss_func = nn.MSELoss()          # mean squared error
    losses = []
    optimizer = torch.optim.Adam(net.parameters())
    for t in range(num_steps):
        optimizer.zero_grad()         # reset accumulated gradients
        predictions = net(xvals)
        loss_val = loss_func(predictions, targets)
        loss_val.backward()           # compute gradients
        optimizer.step()              # update parameters
        losses.append(loss_val.item())  # .item() extracts the Python float
    return losses
```

### Results

| Attempt | Hidden units | Initialization | Fit quality |
|---|---|---|---|
| 1 | 10 | default | Mediocre — misses the bump near $x = 75$, even though the loss has converged |
| 2 | 50 | default | Somewhat better, still misses details |
| 3 | 10 | biases scaled by 80 | Much better |
| 4 | 50 | biases scaled by 80 | Near-perfect |

The modified initialization:

```python
class SingleLayerNetTryTwo(nn.Module):
    def __init__(self, layer_size):
        super(SingleLayerNetTryTwo, self).__init__()
        self.Linear1 = nn.Linear(1, layer_size)
        # Spread the biases widely so the sigmoid "steps" land across [-100, 100]
        with torch.no_grad():
            self.Linear1.bias *= 80
        self.activation = nn.Sigmoid()
        self.Linear2 = nn.Linear(layer_size, 1)
    # forward() unchanged
```

**Why it helps:** each hidden unit's sigmoid transitions at $x = -b_i / w_i$. With default (small) biases, all transitions cluster near $x = 0$, so the net cannot place steps at $x = \pm 50$ or $75$. Scaling the biases spreads the transition points across the input range.

### Lessons learned

- Having enough **capacity** does not mean the network will **find** the function.
- **Overparameterizing** (more units) can make hard functions easier to learn.
  - This can also cause **overfitting** — check for it!
- **Initialization and data normalization matter a lot.**
  - Knowledge about your data can help your model train.

---

## 3. Multi-Layer Perceptron (MLP)

Single-layer perceptron, compact form:

```
x ──► [ sigmoid block: M1, b ] ──► [ linear block: M2 ] ──► y
```

An MLP repeats the nonlinear block several times:

```
x ──► [σ block: M1, b1] ──► [σ block: M2, b2] ──► [σ block: M3, b3] ──► [linear block: M4] ──► y
```

The number of stacked blocks is the **depth**. "Deep learning" means using deep architectures.

Coordinate-wise, one layer computes

```math
h^{(\ell)}_i = \sigma\left( \sum_j M_\ell[i, j] \; h^{(\ell - 1)}_j + b_\ell[i] \right)
```

---

## 4. Activation Functions

### Sigmoid

```math
\sigma(x) = \frac{\exp(x)}{1 + \exp(x)} = \frac{1}{1 + \exp(-x)}
```

No longer popular because of **vanishing gradients**.

### Vanishing gradients

- Gradient descent is a **local search** algorithm.
- A small gradient means changing $\theta$ a little barely changes the loss $\mathcal{L}_S(\theta)$, so SGD gets stuck or moves very slowly on flat regions.
- The sigmoid's derivative is $\sigma'(x) = \sigma(x)(1 - \sigma(x)) \le 1/4$, and it is nearly $0$ in the **saturation regions** (roughly $|x| > 5$). Learning only happens in a narrow range around $0$.
- Through backpropagation, gradients are multiplied across layers; with $L$ sigmoid layers the factor can shrink like $(1/4)^L$. **The deeper the network, the more likely it hits saturation**, so deep sigmoid networks are hard to train.

### ReLU (Rectified Linear Unit)

```math
\sigma(x) = \max(x, 0)
```

- Shares the sigmoid's idea of being "off" for negative inputs.
- Does **not saturate** on the positive side (derivative is exactly $1$ for $x > 0$), so it has a much larger range of "learning".
- A ReLU network is **piecewise linear**.

Modern MLP:

```
x ──► [ReLU block] ──► [ReLU block] ──► [ReLU block] ──► [linear block] ──► y
```

### Variations on ReLU

| Name | Formula | Notes |
|---|---|---|
| Softplus | $\sigma(x) = \log(1 + \exp(x))$ | "Smooth ReLU"; its derivative is the sigmoid |
| GELU | $\sigma(x) = \dfrac{x}{2} \left(1 + \mathrm{erf}\left(\dfrac{x}{\sqrt{2}}\right)\right) = x \, \Phi(x)$ | Much more recent; $\Phi$ is the standard normal CDF; slightly negative dip near $x \approx -0.75$ |

---

## 5. Loss Functions for Classification

### Setup

A classification network outputs a score vector $p \in \mathbb{R}^C$ (the last layers are often called the **head**):

```
x ──► [ReLU: M1, b1] ──► [ReLU: M2, b2] ──► [ReLU: M3, b3] ──► [linear: M4] ──► p ∈ R^C
```

Prediction = index with the highest score:

```math
\text{prediction} = \arg\max_j \, p[j]
```

Dataset $S = \lbrace (x_1, y_1), \ldots, (x_N, y_N) \rbrace$ with **one-hot** labels $y_i \in \mathbb{R}^C$:

```math
y_i = (0, \ldots, 1, \ldots, 0), \qquad y_i[k] = 1 \text{ only if } x_i \text{ has label } k
```

> **Example (watch movements, $C = 3$):** classes = (manual, automatic, quartz). A photo of a caliber with a visible rotor would have label $y = (0, 1, 0)$.

### True (0-1) classification loss

We really want to minimize the number of training mistakes:

```math
\sum_{\text{dataset}} \mathbb{1}[\text{prediction} \ne \text{label}], \qquad \ell(y, p) = 1 - y[\text{prediction}]
```

**Why we can't optimize it directly:**

- It is **non-differentiable** (a step function of $p_{\text{correct}} - p_{\text{incorrect}}$).
- Even where differentiable, its gradient is $0$ everywhere — the ultimate **vanishing gradient**.

So we use **surrogate losses**.

### Hinge loss

> Hinge is to the true loss as ReLU is to the sigmoid.

```math
\text{hinge}(y, p) = \max\left( \max_{i \ne \text{correct}} p_i - p_{\text{correct}} + 1, \; 0 \right)
```

- The true-label score must beat **every** other score by a margin of $1$ to get zero loss.
- Binary form (labels $y_i \in \lbrace -1, +1 \rbrace$): $\max(0, 1 - y_i f(x_i))$, versus the 0-1 loss $\mathbb{1}[y_i f(x_i) < 0]$.
- Hinge is the **smallest convex upper bound** of the 0-1 loss.

**Worked example:** scores $p = (0.7, 0.15, 0.1, 0.05)$, true class is class 2, so $y = (0, 1, 0, 0)$ and $p_{\text{correct}} = 0.15$:

```math
\text{hinge}(y, p) = \max(0.7 - 0.15 + 1, \; 0) = 1.55
```

#### Hinge vs. true loss — quiz questions

Whenever the prediction is wrong, $p_{\text{correct}} \le \max_{i \ne \text{correct}} p_i$, so hinge $\ge 1$. Hence

```math
\mathbb{1}[\text{mistake}] \le \text{hinge}(y, p)
```

- Average hinge loss $= 0.1$ ⟹ at most **10%** of examples are misclassified.
- Average hinge loss $= 0.99$ ⟹ at most **99%** are misclassified — nearly useless as a guarantee.

### Cross-entropy loss

Inputs:

1. A distribution over the $C$ classes: $p = (p_1, \ldots, p_C)$ with $\sum_{i=1}^{C} p_i = 1$.
2. The true class as a one-hot vector $y$.

```math
\ell(p, y) = \sum_i -\mathbb{1}[y = i] \log p_i = -\log p_{\text{correct class}}
```

Behavior:

- $p_{\text{correct}} = 1 \Rightarrow \ell = -\log 1 = 0$
- $p_{\text{correct}} \to 0 \Rightarrow \ell \to \infty$

#### Relation to the 0-1 loss

Cross-entropy is convex (in the logits, after softmax) and non-negative. How does it bound the 0-1 loss?

- We **don't** choose class $k$ when $k \ne \arg\max_i p_i$.
- If $k$ is not the argmax, some $p_j \ge p_k$, so $2 p_k \le p_k + p_j \le 1$, i.e. $p_k \le \tfrac{1}{2}$.
- So when the prediction is **wrong**: $\ell(p, y) = -\log p_k \ge -\log \tfrac{1}{2} = \log 2$.
- When the prediction is **right**: $\ell(p, y) \ge 0 = \mathbb{1}[\text{mistake}]$.

Therefore, in general:

```math
\ell(p, y) \ge \log 2 \cdot \mathbb{1}[\text{prediction} \ne \text{true label}]
```

Equivalently, the fraction of mistakes is at most the average cross-entropy divided by $\log 2$ (which is exactly $1$ if you use $\log_2$, i.e. bits).

---

## 6. Softmax and Its Numerical Stability

Cross-entropy needs a **probability distribution**, but a linear layer's output is never normalized. **Softmax** converts any vector into one:

```math
\mathrm{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}, \qquad \sum_i \mathrm{softmax}(x)_i = 1
```

> **Example (soccer):** a model predicting a match outcome (home win, draw, away win) produces logits $(1, 2, 3)$. Softmax gives $(0.090, 0.245, 0.665)$ — a 66.5% chance of an away win.

### The naive implementation breaks

```python
import numpy as np

def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

x = softmax([1, 2, 3])
y = softmax([1000, 2000, 3000])
print('x = ', x)   # [0.09003057 0.24472847 0.66524096]
print('y = ', y)   # [nan nan nan]
```

`np.exp(1000)` overflows to `inf`, and `inf / inf = nan`.

### The fix: shift by a constant

Softmax is invariant to adding the same constant to every input. For any $C > 0$:

```math
s_i = \frac{e^{a_i}}{\sum_{j=1}^{N} e^{a_j}} = \frac{C e^{a_i}}{\sum_{j=1}^{N} C e^{a_j}} = \frac{e^{a_i + \log C}}{\sum_{j=1}^{N} e^{a_j + \log C}}
```

Choose $\log C = -\max_j a_j$. Then the largest exponent is $e^0 = 1$, so nothing overflows:

```python
def stable_softmax(x):
    x = np.asarray(x, dtype=float)
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

stable_softmax([1000, 2000, 3000])   # [0. 0. 1.]
```

The same idea gives the stable **log-sum-exp** used inside cross-entropy:

```math
\log \sum_j e^{a_j} = m + \log \sum_j e^{a_j - m}, \qquad m = \max_j a_j
```

---

## 7. Cross-Entropy and KL-Divergence

**Entropy** of a distribution $\pi$:

```math
H(\pi) = -\sum_i \pi_i \log \pi_i
```

**KL-divergence** from $\pi$ to $p$:

```math
D(\pi \Vert p) = \sum_i \pi_i \log \frac{\pi_i}{p_i}
```

- $D(\pi \Vert p) \ge 0$, and $D(\pi \Vert p) = 0$ only if $\pi = p$.

Treat the label $y = \pi$ as a distribution. Then

```math
\mathrm{crossentropy}(p, y) = -\sum_i y_i \log p_i = H(y) + D(y \Vert p)
```

For a one-hot $y$, $H(y) = 0$, so minimizing cross-entropy is exactly minimizing the KL-divergence between the label distribution and the model's prediction. This form also works for **soft labels** (e.g. label smoothing), where $H(y) > 0$ is a constant that doesn't affect the gradient.

---

## 8. Summary: The Standard Architecture

```
x ──► [ReLU block] ──► [ReLU block] ──► [linear block] ──► [softmax block] ──► p ∈ R^C
```

- **Classification:** ReLU blocks, then a linear block, then softmax + cross-entropy.
- **Regression:** typically drop the softmax and use a loss like mean squared error on the linear output.
- **Other tasks:** swap in the appropriate head and loss.

**Deep learning is modular** — like LEGO bricks, blocks snap together in different combinations.

---

## 9. Short Intro to PyTorch

### Why use a deep learning library?

- Architectures (VGG, ResNet, Bi-LSTM, …) get very complicated.
- Instead of hand-writing gradient code, use an **autograd** library.
- You specify the **computation graph**; the library finds gradients.

Popular libraries: **PyTorch**, **JAX**, **TensorFlow**, Keras; older ones include Caffe/Caffe2, MXNet, CNTK, Chainer, Theano, and Lua Torch.

### A two-layer network

```python
class TwoLayerNet(torch.nn.Module):
    def __init__(self, input_size, layer_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.Linear1 = torch.nn.Linear(input_size, layer_size)
        self.activation1 = torch.nn.ReLU()
        self.Linear2 = torch.nn.Linear(layer_size, output_size)
        self.activation2 = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        out = self.Linear1(x)
        out = self.activation1(out)
        out = self.Linear2(out)
        out = self.activation2(out)
        out = self.softmax(out)
        return out
```

### Tensors and batches

```python
x = torch.tensor([[ 1.0, 1.0],
                  [ 1.0, 0.0],
                  [-1.0, 2.0],
                  [ 2.0, 4.0]])   # batch size = 4, each point is 2-D

y = torch.tensor([[0, 0, 1],
                  [1, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0]])     # one-hot labels, 3 classes
```

- A **tensor** is an $n$-dimensional array.
- By convention, the **first dimension is the batch dimension**.

### Computing the loss

```python
net = TwoLayerNet(2, 10, 3)
scores = net(x)
loss_func = torch.nn.CrossEntropyLoss()
label_indices = np.argmax(y, axis=1)     # tensor([2, 0, 0, 1])
loss = loss_func(scores, label_indices)  # e.g. tensor(1.0730)
```

Note: `CrossEntropyLoss` takes **class indices**, not one-hot vectors.

### A general multi-layer network

```python
class MultiLayerNet(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(MultiLayerNet, self).__init__()
        layer_sizes = [input_size] + layer_sizes
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1])
             for i in range(len(layer_sizes) - 1)])
        self.activations = torch.nn.ModuleList(
            [torch.nn.ReLU() for _ in self.linear_layers])
        self.final_linear = torch.nn.Linear(layer_sizes[-1], output_size)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, x):
        out = x
        for linear_layer, activation in zip(self.linear_layers, self.activations):
            out = linear_layer(out)
            out = activation(out)
        out = self.final_linear(out)
        out = self.softmax(out)
        return out
```

### ⚠️ Practical pitfalls (not on the slides, but worth knowing)

1. **Use `nn.ModuleList`, not a plain Python list.** The slide version stores layers in a list, so `net.parameters()` won't see them and the optimizer will never update them. (Fixed above.)
2. **`CrossEntropyLoss` already applies log-softmax.** It expects raw **logits**. Feeding it softmax outputs applies softmax twice, which squashes gradients. For training, drop the final `Softmax` and apply it only when you need probabilities.
3. **Don't put ReLU right before the output.** `TwoLayerNet` applies ReLU to the final logits, which forces all scores to be $\ge 0$ and can zero out classes. The standard head is just a linear layer.

---

## 10. Next Time

- The **backpropagation** algorithm
- How to build an automatic differentiation package
- What those "graph" and "leaf" messages from PyTorch/TensorFlow actually mean
