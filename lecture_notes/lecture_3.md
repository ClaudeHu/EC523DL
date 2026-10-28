# Math/ML Review II — Generalization, Gradient Descent and SGD

> Notes based on *Deep Learning 2023, Brian Kulis* (EC523 lecture slides).

---

## Table of Contents

1. [Review: Supervised Learning](#1-review-supervised-learning)
2. [Linear Regression (closed form)](#2-linear-regression-closed-form)
3. [Logistic Regression (no closed form)](#3-logistic-regression-no-closed-form)
4. [Gradient Descent (GD)](#4-gradient-descent-gd)
5. [Stochastic Gradient Descent (SGD)](#5-stochastic-gradient-descent-sgd)
6. [GD vs SGD](#6-gd-vs-sgd)
7. [Generalization and Overfitting](#7-generalization-and-overfitting)
8. [Combatting Overfitting](#8-combatting-overfitting)
9. [Cheat Sheet](#9-cheat-sheet)

---

## 1. Review: Supervised Learning

**Pipeline**

1. Gather training data

```math
S_{\text{train}} = \lbrace (x_1, y_1), \dots, (x_N, y_N) \rbrace
```

2. Choose a parametrized model $f : X \times \Theta \to Y$
3. **Train**: find $\hat{\theta} \in \Theta$ such that $f(x, \hat{\theta}) \approx y$
4. Test on a new, held-out test set

**How to do step 3 (training)?**

| Approach | When to use | Objective |
| --- | --- | --- |
| **Maximum Likelihood Estimation (MLE)** | $f(x, \theta)$ is a parametrized family of probability distributions | Maximize probability of the training data |
| **Empirical Risk Minimization (ERM)** | Any model with a chosen loss $\ell(x, y, \theta)$ | Minimize the average training loss |

---

## 2. Linear Regression (closed form)

Model for a 1D feature: $y \approx \theta_1 + \theta_2 x$

**Loss**

```math
L_S(\theta) = \frac{1}{2} \sum_{(x,y) \in S} \big( y - (\theta_1 + \theta_2 x) \big)^2
```

**Gradients**

```math
\frac{\partial L_S(\theta)}{\partial \theta_1} = - \sum_{(x,y) \in S} \big( y - (\theta_1 + \theta_2 x) \big)
```

```math
\frac{\partial L_S(\theta)}{\partial \theta_2} = - \sum_{(x,y) \in S} x \big( y - (\theta_1 + \theta_2 x) \big)
```

**Closed-form solution** (set gradients to zero). Define the sample averages

```math
\bar{y} := \frac{1}{\lvert S \rvert} \sum_{S} y, \qquad
\bar{x} := \frac{1}{\lvert S \rvert} \sum_{S} x, \qquad
\overline{x^2} := \frac{1}{\lvert S \rvert} \sum_{S} x^2
```

Then

```math
\hat{\theta}_2 = \frac{\frac{1}{\lvert S \rvert} \sum_{S} x (y - \bar{y})}{\overline{x^2} - \bar{x}^2}
= \left( 1 - \frac{\bar{x}^2}{\overline{x^2}} \right)^{-1} \frac{1}{\lvert S \rvert \overline{x^2}} \sum_{S} x (y - \bar{y})
```

```math
\hat{\theta}_1 = \bar{y} - \hat{\theta}_2 \bar{x}
```

> The loss surface is a convex "bowl" (quadratic) in $(\theta_1, \theta_2)$, so there is a unique minimum.

---

## 3. Logistic Regression (no closed form)

- Binary classification: $y \in \lbrace -1, +1 \rbrace$
- Logistic model:

```math
p(y \mid x; \theta) \propto \exp\big( y (\theta_1 + \theta_2 x) \big)
\quad \Longrightarrow \quad
p(y \mid x; \theta) = \frac{1}{1 + \exp\big( -y (\theta_1 + \theta_2 x) \big)}
```

- Logistic loss (negative log-likelihood):

```math
\ell(x, y, \theta) = -\log p(y \mid x; \theta) = \log\Big( 1 + \exp\big( -y(\theta_1 + \theta_2 x) \big) \Big)
```

```math
L_S(\theta) = \frac{1}{\lvert S \rvert} \sum_{(x,y) \in S} \ell(x, y, \theta)
```

- **There is no closed-form solution** for $\operatorname{argmin}_\theta L_S(\theta)$ → we need an iterative method such as gradient descent.

---

## 4. Gradient Descent (GD)

### 4.1 Idea

- "Stochastic" = random. Plain GD is **deterministic**.
- GD is much slower than SGD on large datasets and is rarely used today, but it is easier to analyze, so we study it first.
- GD is a **local search**: start at an initial guess $\theta_0$, then produce $\theta_1, \theta_2, \dots, \theta_T = \hat{\theta}$, each one hopefully better than the last.

### 4.2 What are we actually looking for?

- Finding $\operatorname{argmin}_\theta L_S(\theta)$ can be **NP-hard** in general.
- If $L_S$ is **convex**, GD finds the argmin (roughly as fast as possible). Classical models (e.g. SVMs) give convex losses.
- Deep neural network losses are almost always **non-convex**.
- So instead we look for a **first-order stationary point**: a $\theta$ where $\lVert \nabla L_S(\theta) \rVert$ is very small.
  - This is *necessary* but *not sufficient* for being the true argmin.

### 4.3 Key assumptions

1. $L_S(\theta) \ge 0$ for all $\theta$ (basically always true).
2. **Bounded second derivative (smoothness):**

```math
\lVert \nabla^2 L_S(\theta) \rVert_{op} \le M \quad \text{for all } \theta
```

- $\nabla^2 f$ is the **Hessian** (matrix of second partial derivatives).
- The operator norm equals the largest eigenvalue (in absolute value) for a symmetric matrix.
- $M$ measures the **"difficulty"** of minimizing $L_S$.
- In practice this often holds only for *most* $\theta$, or only with a very large $M$.

### 4.4 Smoothness parameter $M$

First-order (linear) approximation around $\theta_t$:

```math
L_S(\theta_t + \delta) \approx L_S(\theta_t) + \langle \nabla L_S(\theta_t), \delta \rangle
```

$M$ tells us how large $\delta$ can be before this approximation becomes bad. By Taylor's theorem, for some $\theta'$ on the segment $[\theta_t, \theta_t + \delta]$:

```math
\begin{aligned}
L_S(\theta_t + \delta)
&= L_S(\theta_t) + \langle \nabla L_S(\theta_t), \delta \rangle + \frac{1}{2} \delta^\top \nabla^2 L_S(\theta') \delta \\
&= L_S(\theta_t) + \langle \nabla L_S(\theta_t), \delta \rangle + O\big( M \lVert \delta \rVert^2 \big)
\end{aligned}
```

### 4.5 The quadratic upper bound $U$

Define

```math
U(\theta_t + \delta) = L_S(\theta_t) + \langle \nabla L_S(\theta_t), \delta \rangle + \frac{M}{2} \lVert \delta \rVert^2
```

Because $\delta^\top \nabla^2 L_S(\theta') \delta \le M \lVert \delta \rVert^2$:

```math
L_S(\theta_t + \delta) \le U(\theta_t + \delta)
```

- $U$ touches $L_S$ at $\theta_t$: $U(\theta_t) = L_S(\theta_t)$.
- $U$ lies above $L_S$ everywhere.
- So if we set $\theta_{t+1} = \operatorname{argmin} U(\theta)$, then

```math
L_S(\theta_{t+1}) \le U(\theta_{t+1}) \le U(\theta_t) = L_S(\theta_t)
```

> **GD guarantees the loss never increases.** (If $M$ is set large enough, it bounds the Hessian, and the decrease is guaranteed.)

### 4.6 Computing the GD update

Minimize $U$ over $\delta$:

```math
\nabla_\delta U(\theta_t + \delta) = \nabla L_S(\theta_t) + M \delta = 0
\quad \Longrightarrow \quad
\delta^\star = -\frac{1}{M} \nabla L_S(\theta_t)
```

**GD update rule:**

```math
\boxed{\theta_{t+1} = \theta_t - \frac{1}{M} \nabla L_S(\theta_t)}
```

The learning rate is $\eta = 1/M$.

### 4.7 How fast does GD make progress? (proof sketch)

Plug $\delta^\star$ into $U$:

```math
\begin{aligned}
L_S(\theta_{t+1}) \le U(\theta_t + \delta^\star)
&= L_S(\theta_t) + \langle \nabla L_S(\theta_t), \delta^\star \rangle + \frac{M}{2} \lVert \delta^\star \rVert^2 \\
&= L_S(\theta_t) - \frac{1}{M} \lVert \nabla L_S(\theta_t) \rVert^2 + \frac{1}{2M} \lVert \nabla L_S(\theta_t) \rVert^2 \\
&= L_S(\theta_t) - \frac{1}{2M} \lVert \nabla L_S(\theta_t) \rVert^2
\end{aligned}
```

Apply repeatedly (telescoping):

```math
L_S(\theta_T) \le L_S(\theta_0) - \frac{1}{2M} \sum_{t=0}^{T-1} \lVert \nabla L_S(\theta_t) \rVert^2
```

Rearrange and use $L_S(\theta_T) \ge 0$:

```math
\sum_{t=0}^{T-1} \lVert \nabla L_S(\theta_t) \rVert^2 \le 2M \big( L_S(\theta_0) - L_S(\theta_T) \big) \le 2M L_S(\theta_0)
```

- The left side is a **convergent sum** bounded by a **constant**, so the terms must decay to zero.

Divide by $T$:

```math
\frac{1}{T} \sum_{t=0}^{T-1} \lVert \nabla L_S(\theta_t) \rVert^2 \le \frac{2M L_S(\theta_0)}{T}
```

The minimum is at most the average, so

```math
\min_{t < T} \lVert \nabla L_S(\theta_t) \rVert \le O\left( \frac{1}{\sqrt{T}} \right)
```

### 4.8 GD summary

| Property | GD |
| --- | --- |
| Update | $\theta_{t+1} = \theta_t - \frac{1}{M} \nabla L_S(\theta_t)$ |
| Guarantee after $T$ steps | $\lVert \nabla L_S \rVert \le O(1/\sqrt{T})$ (squared norm $\le O(1/T)$) |
| Cost per iteration | $O(\lvert S \rvert)$ — a full pass over the dataset |
| Monotone? | Yes, loss decreases every step |

---

## 5. Stochastic Gradient Descent (SGD)

### 5.1 Why SGD?

- Only $O(1)$ memory overhead for large datasets.
- Relatively easy to code.
- In many settings, SGD is in some sense the **fastest** algorithm for finding a minimum.
- **Anytime algorithm**: gives successively better solutions, so you can stop at any time and get a reasonable answer.

### 5.2 From deterministic to stochastic

The loss is an average over data:

```math
L_S(\theta) = \frac{1}{\lvert S \rvert} \sum_{S} \ell(x, y, \theta)
```

Each GD iteration costs $O(\lvert S \rvert)$. **Add randomization:**

1. Choose a datapoint $(x, y) \in S$ uniformly at random.
2. Update

```math
\boxed{\theta_{t+1} = \theta_t - \eta \nabla_\theta \ell(x, y, \theta_t)}
```

### 5.3 The stochastic gradient is unbiased

```math
\mathbb{E}\big[ \nabla_\theta \ell(x, y, \theta) \big]
= \frac{1}{\lvert S \rvert} \sum_{S} \nabla_\theta \ell(x, y, \theta)
= \nabla_\theta \frac{1}{\lvert S \rvert} \sum_{S} \ell(x, y, \theta)
= \nabla_\theta L_S(\theta)
```

Therefore, in expectation, one SGD step equals one GD step:

```math
\mathbb{E}[\theta_{t+1}] = \theta_t - \eta \mathbb{E}\big[ \nabla_\theta \ell(x, y, \theta_t) \big] = \theta_t - \eta \nabla L_S(\theta_t)
```

### 5.4 Key assumptions for SGD

1. $L_S(\theta) \ge 0$ for all $\theta$.
2. $\lVert \nabla^2 L_S(\theta) \rVert_{op} \le M$ for all $\theta$.
3. **Bounded stochastic gradients:**

```math
\lVert \nabla_\theta \ell(x, y, \theta) \rVert \le G \quad \text{for all } x, y, \theta
```

- $G$ is another measure of difficulty: it bounds the amount of **"noise"** in the problem.
- Because gradients are noisy, we trust them less by using a **small learning rate** $\eta$.

### 5.5 How fast does SGD make progress? (proof sketch)

Same upper bound $U$, now with $\delta = -\eta \nabla \ell(x, y, \theta_t)$:

```math
U(\theta_{t+1}) = L_S(\theta_t) - \eta \langle \nabla L_S(\theta_t), \nabla \ell(x, y, \theta_t) \rangle + \frac{M \eta^2}{2} \lVert \nabla \ell(x, y, \theta_t) \rVert^2
```

Take expectation over the random datapoint (unbiasedness + bounded gradients):

```math
\mathbb{E}\big[ U(\theta_{t+1}) \big] \le L_S(\theta_t) \underbrace{- \eta \lVert \nabla L_S(\theta_t) \rVert^2}_{\text{progress}} + \underbrace{\frac{M \eta^2 G^2}{2}}_{\text{movement penalty}}
```

Telescope over $T$ steps:

```math
\mathbb{E}\big[ L_S(\theta_T) \big] \le L_S(\theta_0) - \eta \sum_{t} \mathbb{E}\big[ \lVert \nabla L_S(\theta_t) \rVert^2 \big] + \frac{T M \eta^2 G^2}{2}
```

Rearrange and use $L_S \ge 0$:

```math
\frac{1}{T} \sum_{t} \mathbb{E}\big[ \lVert \nabla L_S(\theta_t) \rVert^2 \big] \le \frac{L_S(\theta_0)}{\eta T} + \frac{M \eta G^2}{2}
```

- First term wants **large** $\eta$; second term wants **small** $\eta$ → trade-off.

Choose $\eta = \dfrac{1}{G \sqrt{T}}$:

```math
\frac{1}{T} \sum_{t} \mathbb{E}\big[ \lVert \nabla L_S(\theta_t) \rVert^2 \big] \le \frac{G L_S(\theta_0)}{\sqrt{T}} + \frac{M}{2 G \sqrt{T}} = O\left( \frac{1}{\sqrt{T}} \right)
```

Taking square roots (Jensen):

```math
\mathbb{E}\big[ \lVert \nabla L_S(\theta_t) \rVert \big] \le O\left( \frac{1}{T^{1/4}} \right)
```

### 5.6 SGD summary

| Property | SGD |
| --- | --- |
| Update | $\theta_{t+1} = \theta_t - \eta \nabla_\theta \ell(x, y, \theta_t)$ |
| Learning rate | $\eta = 1/(G \sqrt{T})$ in theory; in practice often guessed |
| Guarantee after $T$ steps | $\mathbb{E}\lVert \nabla L_S \rVert \le O(T^{-1/4})$ (squared norm $\le O(1/\sqrt{T})$) |
| Cost per iteration | $O(1)$ |
| Monotone? | No, some steps can make negative progress |

> ⚠️ Note: one slide writes the SGD update with a "+" sign. It should be a **minus** (we move *against* the gradient).

---

## 6. GD vs SGD

| | GD | SGD |
| --- | --- | --- |
| Time per step | $O(\lvert S \rvert)$ | $O(1)$ |
| Every step improves $L_S$? | Yes | No |
| Gradient size after $T$ steps | $O(1/\sqrt{T})$ | $O(T^{-1/4})$ (in expectation) |

**Fair comparison: fix total compute budget $N$.**

- GD can take $T = N / \lvert S \rvert$ steps:

```math
\lVert \nabla L_S \rVert \le O\left( \sqrt{\frac{\lvert S \rvert}{N}} \right)
```

- SGD can take $T = N$ steps:

```math
\mathbb{E} \lVert \nabla L_S \rVert \le O\left( \frac{1}{N^{1/4}} \right)
```

- Crossover when $\sqrt{\lvert S \rvert / N} = N^{-1/4}$, i.e.

```math
N = \lvert S \rvert^2
```

> SGD wins unless you can afford roughly $\lvert S \rvert$ full passes over the data. And even then, GD's "win" is mostly an illusion caused by overfitting (next section).

### Implementation notes

- In practice $M$ and $G$ are unknown, so the learning rate is set by heuristics.
- Instead of sampling a random datapoint each step, most implementations **shuffle** $S$ and iterate through it in that order (one pass = one **epoch**). This tends to work better in practice, though the theory is less well understood.

---

## 7. Generalization and Overfitting

- We **train** by minimizing $L_{S_{\text{train}}}(\theta)$, but we actually **care about** $L_{S_{\text{test}}}(\theta)$.
- These differ when data is not i.i.d., and also simply due to **ordinary sampling error**.

### 7.1 Overfitting as multiple hypothesis testing (the "expert" example)

**Setup**

- Predict whether the NASDAQ rises or falls each day.
- Poll **10,000 experts** on **10 days**; each says "yes" or "no".
- **Alice** happens to be correct on all 10 days → ERM says: "always follow Alice."

**Reality**

- Only **Barb** is good: correct with probability $9/10$ (she made her expected 1 mistake in 10 days).
- Everyone else (including Alice) flips a fair coin.
- A single coin-flipper is perfect on 10 days with probability $1/2^{10} = 1/1024$.

**But with many experts, someone will look perfect:**

```math
P(\text{some coin-flipper is perfect}) = 1 - \left( \frac{1023}{1024} \right)^{9999} > 0.99
```

> This is a **failure of ERM**: the training performance (perfect) is very different from the test performance (50%).

### 7.2 Ways to avoid the problem

1. **Gather more training data** (more than 10 days).
2. **Remove some experts at random** (!). Throw out $9/10$ of them, leaving about 1000. Barb survives with probability $1/10$, so

```math
P(\text{not picking Barb}) \approx 1 - \frac{1}{10} \left( \frac{1023}{1024} \right)^{1000} \approx 0.96
```

Still bad here, but it shows that **fewer hypotheses → less chance of a fluke winner**.

### 7.3 Connection to ML

- Each parameter value $\theta \in \Theta$ is like an **"expert"**.
- If we check **too many** values of $\theta$, or $S_{\text{train}}$ is **too small**, some bad $\theta$ can look great on the training set but perform poorly on the test set.
- Typical training curve as SGD iterations increase (i.e. more $\theta$ values tested):
  - $L_{S_{\text{train}}}(\theta)$ keeps decreasing.
  - $L_{S_{\text{test}}}(\theta)$ decreases, then **starts increasing** → overfitting.

### 7.4 How much data do you need? (biased coin)

- Estimate the probability a biased coin lands heads from $N$ flips.
- $F_i = 1$ if flip $i$ is heads, $0$ otherwise.

```math
A = \frac{1}{N} (F_1 + \dots + F_N), \qquad \operatorname{Var}(F_i) \le 1
```

- With probability at least $0.99$ (Chebyshev):

```math
\lvert A - \mathbb{E}[F] \rvert \le \frac{10}{\sqrt{N}}
```

- **General rule:** with $N$ datapoints you usually cannot estimate quantities to better than

```math
O\left( \frac{1}{\sqrt{N}} \right)
```

### 7.5 Two myths about overfitting

1. **Myth:** "The complexity term $C$ equals the dimension of $\Theta$."
   - $C$ measures the *size* of $\Theta$, but there are much better ways to measure size.
   - Many take the form $C = \max_{\theta \in \Theta} \lVert \theta \rVert$ for some norm other than the standard one.
   - Finding the "right" $C$ is an active research area.
2. **Myth:** "Deep neural networks never overfit."
   - They often overfit *less* than other models, but they **do** overfit sometimes. **Always check.**

---

## 8. Combatting Overfitting

1. **Gather more data!** (the best fix)
2. Everything else is a more sophisticated version of **"throw out some of the experts"**, and these methods work surprisingly well.

### 8.1 Early stopping

- Monitor validation/test loss during training.
- **Stop training** at the point where $L_{S_{\text{test}}}$ is lowest, before it starts rising.

### 8.2 Regularization

Minimize a **penalized** empirical risk instead of the raw one:

```math
L(\theta) = L_S(\theta) + r(\theta)
```

- $r(\theta)$ is the **regularizer**.
- Common choice — **weight decay** ($L_2$ regularization):

```math
r(\theta) = \lambda \lVert \theta \rVert^2
```

- $\lambda$ is a **hyperparameter** chosen by the user.
- Intuition: "throw out all large $\theta$ unless they have really good training error."

Effect on the GD/SGD update (gradient of $\lambda \lVert \theta \rVert^2$ is $2 \lambda \theta$):

```math
\theta_{t+1} = \theta_t - \eta \big( \nabla \ell(x, y, \theta_t) + 2 \lambda \theta_t \big) = (1 - 2 \eta \lambda) \theta_t - \eta \nabla \ell(x, y, \theta_t)
```

The weights literally "decay" by a factor $(1 - 2\eta\lambda)$ each step.

---

## 9. Cheat Sheet

| Concept | Formula / Fact |
| --- | --- |
| GD update | $\theta_{t+1} = \theta_t - \frac{1}{M} \nabla L_S(\theta_t)$ |
| GD per-step decrease | $L_S(\theta_{t+1}) \le L_S(\theta_t) - \frac{1}{2M} \lVert \nabla L_S(\theta_t) \rVert^2$ |
| GD rate | $\min_t \lVert \nabla L_S(\theta_t) \rVert \le O(1/\sqrt{T})$ |
| SGD update | $\theta_{t+1} = \theta_t - \eta \nabla_\theta \ell(x, y, \theta_t)$ |
| SGD learning rate (theory) | $\eta = 1/(G \sqrt{T})$ |
| SGD rate | $\mathbb{E} \lVert \nabla L_S \rVert \le O(T^{-1/4})$ |
| Smoothness | $\lVert \nabla^2 L_S(\theta) \rVert_{op} \le M$ |
| Bounded noise | $\lVert \nabla_\theta \ell(x, y, \theta) \rVert \le G$ |
| GD/SGD crossover | total compute $N = \lvert S \rvert^2$ |
| Estimation error | $O(1/\sqrt{N})$ with $N$ samples |
| Weight decay | $r(\theta) = \lambda \lVert \theta \rVert^2$ |
