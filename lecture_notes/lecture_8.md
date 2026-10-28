# Lecture 8 — Deep Learning: Training II

> Topics: **Initialization** · **Transfer Learning** · **Optimization** (SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW) · **Learning-rate schedules** · **Hyperparameter tuning**

---

## Table of Contents

1. [Weight Initialization](#1-weight-initialization)
   - [Why not zero?](#11-why-not-initialize-to-zero)
   - [Naive random init and exploding activations](#12-naive-random-initialization)
   - [Xavier initialization](#13-xavier-glorot-initialization)
   - [Kaiming initialization](#14-kaiming-he-initialization)
2. [Transfer Learning](#2-transfer-learning)
3. [Optimization](#3-optimization)
   - [Gradient descent and SGD](#31-gradient-descent-vs-sgd)
   - [Second-order methods (Newton)](#32-second-order-methods-newtons-method)
   - [Momentum](#33-sgd-with-momentum)
   - [Nesterov momentum](#34-nesterov-momentum)
   - [Adaptive learning rates](#35-adaptive-learning-rates)
   - [AdaGrad](#36-adagrad)
   - [RMSProp](#37-rmsprop)
   - [Adam](#38-adam)
   - [AdamW](#39-adamw-decoupled-weight-decay)
   - [Comparison table](#310-optimizer-comparison)
4. [Learning-Rate Schedules](#4-learning-rate-schedules)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Summary / Cheat Sheet](#6-cheat-sheet)

---

## 1. Weight Initialization

Network under consideration:

```
x ──► [linear W1] ──► [ReLU W2] ──► [ReLU W3] ──► [ReLU W4] ──► y
```

**Question:** how should we initialize $W_1, W_2, \dots$?

### 1.1 Why not initialize to zero?

Take the simplest case: two linear blocks, no activation.

```
x ──► [linear W1] ──z──► [linear W2] ──► y
```

Forward pass:

$$
z = x W_1 \qquad y = z W_2 = x W_1 W_2
$$

If $W_1 = W_2 = 0$, then $z = 0$, and with $D$ = the downstream gradient $\partial \mathcal{L} / \partial y$:

$$
\frac{\partial \mathcal{L}}{\partial W_2} = z^{\top} D = 0
\qquad
\frac{\partial \mathcal{L}}{\partial W_1} = x^{\top} D W_2^{\top} = 0
$$

➡️ **Both gradients are zero — nothing ever changes.** (More generally, any *constant* init makes all neurons in a layer identical: the symmetry is never broken.)

**Fix:** random initialization — draw each $W[i, j]$ independently from a fixed distribution. But *which* one?

| Candidate | Problem |
|---|---|
| $\mathcal{U}(0, 1)$ | All positive, not centered |
| $\mathcal{N}(0, 1)$ | Variance too large → activations explode |
| $\mathcal{U}(-1, 1)$ | Same issue, scale not tied to layer width |
| $\mathcal{N}(0, \sigma^2)$ | ✅ — need to pick $\sigma^2$ carefully |

### 1.2 Naive random initialization

Initializing every weight with $\mathcal{N}(0, 1)$ gives **unstable predictions**: as the network gets deeper, the activations at each layer either blow up or vanish.

> Goal: make sure that the values of $y$ do **not explode or vanish**, even in very deep networks.

**Why does this happen?** Consider a linear layer with no activation and $N$ inputs:

$$
y[j] = \sum_{i=1}^{N} x[i] \thinspace W[i, j]
$$

Measure the magnitude with $\mathbb{E}\big[y[j]^2\big]$. Assume the $x[i]$ are independent with $\mathbb{E}\big[x[i]^2\big] = \tau^2$, and $W[i, j] \sim \mathcal{N}(0, \sigma^2)$ independent of $x$. Cross terms with $i \neq i'$ vanish since the weights are zero-mean:

$$
\mathbb{E}\big[y[j]^2\big]
= \mathbb{E}\Big[\sum_{i, i'} x[i] \thinspace W[i, j] \thinspace x[i'] \thinspace W[i', j]\Big]
= \sum_{i} \mathbb{E}\big[x[i]^2\big] \thinspace \mathbb{E}\big[W[i, j]^2\big]
= N \tau^2 \sigma^2
$$

- With $\sigma^2 = 1$: $\mathbb{E}[y^2] = N\tau^2$ → **squared magnitude grows by a factor of $N$ every layer** (geometric explosion).
- **Fix:** choose $\sigma^2 = \dfrac{1}{N}$ so that output magnitude = input magnitude.

### 1.3 Xavier (Glorot) Initialization

- Forward pass stability → scale variance by the **number of inputs** $N_{\text{in}}$.
- The backward pass through a linear layer is *also* a linear layer, but with input and output dimensions swapped → gradient stability wants scaling by the **number of outputs** $N_{\text{out}}$.
- Xavier is a **compromise**:

$$
\boxed{\sigma^2 = \frac{2}{N_{\text{in}} + N_{\text{out}}}}
$$

✅ Works well for linear (no-activation) networks.

❌ **Adding back ReLU:** activations shrink layer by layer and vanish in deep nets.

**Why?** The Xavier derivation ignored the activation function, and **ReLU decreases magnitudes**. Each ReLU layer causes a geometric decrease.

**Toy example.** Let $X = +2$ or $-2$ with equal probability:

$$
\mathbb{E}[X^2] = \tfrac{1}{2} \cdot 2^2 + \tfrac{1}{2} \cdot (-2)^2 = 4
$$

$$
\mathbb{E}\big[\mathrm{ReLU}(X)^2\big] = \tfrac{1}{2} \cdot 2^2 + \tfrac{1}{2} \cdot 0^2 = 2
$$

ReLU **halves** the second moment of a symmetric input.

### 1.4 Kaiming (He) Initialization

Redo the calculation for a single layer $y = \mathrm{ReLU}(xW)$:

$$
y[j] = \mathrm{ReLU}\Big(\sum_{i=1}^{N} x[i] \thinspace W[i, j]\Big)
$$

With $W[i, j] \sim \mathcal{N}(0, \sigma^2)$, each $x[i] \thinspace W[i, j]$ is symmetric about zero, so ReLU keeps exactly half the second moment:

$$
\mathbb{E}\big[y[j]^2\big] = \frac{1}{2} \thinspace \mathbb{E}\Big[\Big(\sum_{i=1}^{N} x[i] \thinspace W[i, j]\Big)^2\Big] = \frac{N}{2} \thinspace \sigma^2 \tau^2
$$

Set the factor to 1:

$$
\boxed{\sigma^2 = \frac{2}{N_{\text{in}}}}
$$

### Xavier vs Kaiming

| | Assumes | Variance of $\mathcal{N}(0, \sigma^2)$ |
|---|---|---|
| **Xavier** | No activation (linear) | $\sigma^2 = \dfrac{2}{N_{\text{in}} + N_{\text{out}}}$ |
| **Kaiming** | ReLU activation | $\sigma^2 = \dfrac{2}{N_{\text{in}}}$ (or $\dfrac{2}{N_{\text{out}}}$ — roughly the same effect) |

> 📌 **PyTorch's default initialization is Kaiming.**

---

## 2. Transfer Learning

**Motivation**

- Regularization is partly about dealing with **insufficient data**.
- Humans learn from little data all the time → it must be possible.
- **Hypothesis:** learning from little data is easier if you already "know" something about the data.
- **Strategy:** take a very good model trained on a *related* task (often with lots of data) and use it to boost performance on a small-data task.

**Intermediate representations**

- For an image classifier, the representation right before the final linear layer **must be (nearly) linearly separable** by the classes.
- Hope: intermediate layers encode a **"better representation"** of the data — an abstract *"vector space of ideas"*.

**Example (audio):**

```
Audio signal ──► [ReLU] ──► [ReLU] ──► [ReLU] ──► (vector in R^d) ──► [Linear] ──► Transcription
                                                       │
                     pitch, word length, speaking speed, excitement, ...
```

To solve a **new task** (e.g. *speaker identity*):

```
Audio signal ──► [ReLU] ──► [ReLU] ──► [ReLU] ──► [NEW Linear] ──► Speaker identity
                 └──── import pretrained weights ────┘  └─ train from scratch ─┘
```

**In PyTorch** (e.g. ResNet): the last layer is `model.fc` (fully connected). Replace it:

```python
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights="IMAGENET1K_V1")    # load pretrained model
model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace last linear layer

# Option A — fine-tune ALL parameters
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Option B — train ONLY the new layer (feature extractor)
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-3, momentum=0.9)
```

Reference: <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>

> 🎸 *Analogy:* a classical guitarist who already has solid right-hand technique (the pretrained "feature extractor") picks up a new piece quickly — only the piece-specific part (the new head) has to be learned from scratch.

---

## 3. Optimization

### 3.1 Gradient Descent vs SGD

**Full-batch gradient descent**

$$
\theta_{t+1} = \theta_t - \eta \thinspace \nabla \mathcal{L}(\theta_t)
$$

From the previous lecture: if $\lVert \nabla^2 \mathcal{L}(\theta) \rVert \le M$ and $\eta = \dfrac{1}{M}$, then

$$
\mathcal{L}(\theta_{t+1}) \le \mathcal{L}(\theta_t) - \frac{1}{2M} \lVert \nabla \mathcal{L}(\theta_t) \rVert^2
$$

➡️ **The loss always decreases.**

**Stochastic gradient descent (SGD)**

$$
\theta_{t+1} = \theta_t - \eta \thinspace \nabla \ell(\theta_t; x_t)
$$

- $\theta$: network parameters; $x_t$: the $t$-th training example (or mini-batch); $\eta \in \mathbb{R}$: learning rate.
- Now it is **possible** that $\mathcal{L}(\theta_{t+1}) \ge \mathcal{L}(\theta_t)$.
- Mini-batch gradients are **noisy estimates** of the true gradient → the path zig-zags.

**Problems with plain (S)GD** on an elongated loss surface (ill-conditioned):

- 🐢 **slow progress in shallow dimensions**
- 🔀 **oscillation in steep dimensions**

**Can we do better?**
1. Use second-derivative (Hessian) information?
2. Use the gradient more cleverly?

### 3.2 Second-Order Methods (Newton's Method)

Approximate the loss by a second-order Taylor expansion around $\theta_0$:

$$
\mathcal{L}(\theta) \approx \mathcal{L}(\theta_0) + \langle \nabla \mathcal{L}(\theta_0) \thinspace , \thinspace \theta - \theta_0 \rangle + \frac{1}{2} (\theta - \theta_0)^{\top} \nabla^2 \mathcal{L}(\theta_0) (\theta - \theta_0)
$$

Minimizing this quadratic gives the Newton step:

$$
\theta_1 = \theta_0 - \big[\nabla^2 \mathcal{L}(\theta_0)\big]^{-1} \nabla \mathcal{L}(\theta_0)
$$

(1-D version: $\theta_1 = \theta_0 - \mathcal{L}'(\theta_0) / \mathcal{L}''(\theta_0)$.)

The Hessian entries are

$$
\nabla^2 \mathcal{L}(\theta)[i, j] = \frac{\partial^2 \mathcal{L}(\theta)}{\partial \theta_i \thinspace \partial \theta_j}
$$

**Why it's not used much in deep learning:**

- If $\theta \in \mathbb{R}^d$: the gradient is in $\mathbb{R}^d$, but the Hessian is in $\mathbb{R}^{d \times d}$. For modern nets $d$ is *really, really big*.
- You must **compute** the matrix *and* **invert** it.
- With a fixed time budget you can take many more gradient steps than Hessian steps. In the large-data / large-model regime, **time is the bottleneck** → gradient descent usually wins overall (cf. Bottou & Bousquet, *The Tradeoffs of Large Scale Learning*).
- Active research continues on making second-order methods faster.

### 3.3 SGD with Momentum

**Physics intuition.** Treat $\mathcal{L}(\theta)$ as the **potential energy** (height) of a ball at position $\theta$.

- Force on the ball $\propto -\nabla \mathcal{L}(\theta)$.
- No friction: energy is conserved → the ball rolls down, reaches max velocity at the bottom, and rolls back up the other side.
- With friction: the ball eventually **settles at the lowest point**.
- Very high friction: stops as soon as it reaches the minimum (≈ plain gradient descent).
- Velocity $v = \dfrac{d\theta}{dt}$ is the **momentum**.

**Ravines.** A ravine is where the surface curves much more steeply in one dimension than in another (common near local minima in deep learning).

- SGD oscillates across the narrow ravine — the negative gradient points down the steep walls instead of along the ravine floor.
- Momentum **averages out** the oscillating components and **amplifies the consistent direction** → accelerates toward the optimum. *Directions of slower change are amplified.*

> 🏎️ *Analogy:* an F1 car through a fast sweeping section — the driver doesn't jerk the wheel left/right at every bump (that's plain SGD); the car's momentum carries it smoothly along the racing line, and small lateral corrections average out.

**Update rule (as in lecture):**

$$
v_t = (1 - \beta) \thinspace v_{t-1} - \beta \thinspace \eta \thinspace \nabla \mathcal{L}(\theta_t)
$$

$$
\theta_{t+1} = \theta_t + v_t
$$

- Here $\beta$ plays the role of **friction**: $\beta = 1$ → regular gradient descent (no memory of past velocity).
- Momentum view: SGD as a **damped particle** on the loss surface. Step = velocity (accumulated past gradients) + current negative gradient.

### 3.4 Nesterov Momentum

- Regular momentum ≈ **Euler integration** of the physics ODE.
- Nesterov uses a smarter integration: evaluate the gradient at the **look-ahead point** $\theta_{t-1} + \beta v_{t-1}$:

$$
v_t = \beta \thinspace v_{t-1} - \eta \thinspace \nabla \mathcal{L}(\theta_{t-1} + \beta \thinspace v_{t-1})
$$

$$
\theta_t = \theta_{t-1} + v_t
$$

(Note: in this formula $\beta$ is the momentum coefficient — "how much velocity to keep".)

- Also called **"accelerated gradient descent"** — *accelerated* means "faster than ordinary GD", **not** physical acceleration. Nesterov's original analysis had no physics intuition.
- Momentum (of either kind) **does overshoot** the optimum; less friction (smaller $\beta$ in the first formulation) → more overshoot.

Reference: Sutskever et al., *On the importance of initialization and momentum in deep learning*, ICML 2013 — <http://proceedings.mlr.press/v28/sutskever13.pdf>

### 3.5 Adaptive Learning Rates

**Sampling as noise.** SGD relies on

$$
\mathbb{E}\big[\nabla \ell(\theta_t; x_t)\big] = \nabla \mathcal{L}(\theta_t)
$$

Define the noise $Z_t = \nabla \ell(\theta_t; x_t) - \nabla \mathcal{L}(\theta_t)$. Then $\mathbb{E}[Z_t] = 0$. **High variance of $Z_t$ → harder optimization.**

**"Optimal" SGD learning rate:**

$$
\eta \propto \frac{1}{\sigma \sqrt{T}}
\qquad \text{where} \qquad
\sigma^2 = \mathbb{E}\big[\lVert \nabla \ell(\theta; x) - \nabla \mathcal{L}(\theta) \rVert^2\big]
$$

- $\dfrac{1}{\sqrt{T}}$: from the SGD lectures ($T$ = number of iterations).
- $\dfrac{1}{\sigma}$: more noise → need a smaller learning rate.

**Estimating $\sigma$ on the fly:**

$$
\sigma^2 = \mathbb{E}\big[\text{grad}^2\big] - \big(\mathbb{E}[\text{grad}]\big)^2
$$

Near convergence $\mathbb{E}[\text{grad}] \to 0$ but $\mathbb{E}[\text{grad}^2]$ need not (individual examples still disagree). So approximate $\sigma^2 \approx \mathbb{E}[\text{grad}^2]$:

$$
\sigma \sqrt{T} \approx \sqrt{\sum_{t=1}^{T} \lVert \nabla \ell(\theta_t; x_t) \rVert^2}
$$

This even lets $\sigma$ drift during training.

**Adaptive GD update:**

$$
\eta_t = \frac{\eta_{\text{base}}}{\sqrt{\sum_{\tau=1}^{t} \lVert \nabla \ell(\theta_\tau; x_\tau) \rVert^2}}
\qquad
\theta_{t+1} = \theta_t - \eta_t \thinspace \nabla \ell(\theta_t; x_t)
$$

- $\eta_{\text{base}}$: a scalar you still tune.
- The sum is roughly $O(\sigma^2 t)$ — a small overestimate of the variance.

**Preconditioning.** If gradients have lots of variance in the $y$ direction but little in the $x$ direction, we want to *trust* the $x$-coordinate more → use **per-coordinate** learning rates.

### 3.6 AdaGrad

Per-coordinate learning rate:

$$
\eta_t[i] = \frac{\eta_{\text{base}}}{\sqrt{\sum_{\tau=1}^{t} \big(\nabla \ell(\theta_\tau; x_\tau)[i]\big)^2}}
\qquad
\theta_{t+1}[i] = \theta_t[i] - \eta_t[i] \thinspace \nabla \ell(\theta_t; x_t)[i]
$$

Compact form ($g_t$ = gradient, $G_0 = 0$, $\odot$ = element-wise product):

$$
G_t = G_{t-1} + g_t \odot g_t
\qquad
\theta_{t+1} = \theta_t - \eta \thinspace \frac{g_t}{\sqrt{G_t} + \varepsilon}
$$

- ✅ Helps when some coordinates have much larger variance than others (a bit like batch-norm's effect).
- ⚠️ Can also hurt — effectively many learning rates to "get right".
- ❗ **Common myth:** AdaGrad's denominator approximates the second derivative. Only true in special cases; the **variance argument is always right**.

```python
grad_squared = 0
while True:
    dtheta = compute_gradient(theta)
    grad_squared += dtheta * dtheta
    theta -= learning_rate * dtheta / (np.sqrt(grad_squared) + 1e-7)
```

**Q: What happens to the step size over a long time?**
It **monotonically decreases** (the sum only grows). This is usually too aggressive and **stops learning too early**. E.g. a high-variance region early in training keeps the step tiny even after reaching a low-variance region — we want to **forget stale information**.

### 3.7 RMSProp

**Exponentially weighted moving average (EWMA):**

$$
A_0 = 0 \qquad A_{t+1} = \beta A_t + (1 - \beta) \thinspace x_t
$$

$$
A_{T+1} = (1 - \beta)\big(\beta^{T-1} x_1 + \dots + \beta x_{T-1} + x_T\big)
$$

Recent values get the most weight; old ones decay geometrically.

**RMSProp** replaces AdaGrad's sum with an EWMA ("leaky" second moment):

$$
\eta_t[i] = \frac{\eta_{\text{base}}}{\sqrt{\sum_{\tau=1}^{t} (1 - \beta) \thinspace \beta^{t - \tau} \thinspace \big(\nabla \ell(\theta_\tau; x_\tau)[i]\big)^2}}
$$

```python
# AdaGrad
grad_squared = 0
while True:
    dtheta = compute_gradient(theta)
    grad_squared += dtheta * dtheta
    theta -= learning_rate * dtheta / (np.sqrt(grad_squared) + 1e-7)

# RMSProp  [Tieleman & Hinton, 2012]
grad_squared = 0
while True:
    dtheta = compute_gradient(theta)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dtheta * dtheta
    theta -= learning_rate * dtheta / (np.sqrt(grad_squared) + 1e-7)
```

### 3.8 Adam

**Adaptive Moment Estimation** = momentum (1st moment) + RMSProp (2nd moment).

**Almost-Adam (no bias correction):**

```python
first_moment = 0
second_moment = 0
while True:
    dtheta = compute_gradient(theta)
    first_moment  = beta1 * first_moment  + (1 - beta1) * dtheta            # momentum
    second_moment = beta2 * second_moment + (1 - beta2) * dtheta * dtheta   # AdaGrad / RMSProp
    theta -= learning_rate * first_moment / (np.sqrt(second_moment) + 1e-7)
```

#### Bias correction

The moving averages start at **zero**, so early estimates are **biased toward zero**.

1. Unroll from $m_0 = 0$:

$$
m_t = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \thinspace g_i
$$

2. Take expectations (assume $\mathbb{E}[g_i] \approx \mathbb{E}[g]$) — the weights sum to $1 - \beta_1^t < 1$:

$$
\mathbb{E}[m_t] = (1 - \beta_1) \sum_{i=1}^{t} \beta_1^{t-i} \thinspace \mathbb{E}[g] = (1 - \beta_1^t) \thinspace \mathbb{E}[g]
$$

3. Divide out the factor:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
\qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\qquad \Rightarrow \qquad
\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]
$$

4. **Example at $t = 1$** with $\beta_1 = 0.9$ and $\beta_2 = 0.999$:

$$
m_1 = 0.1 \thinspace g_1 \qquad v_1 = 0.001 \thinspace g_1^2
$$

- Without correction: $\dfrac{m_1}{\sqrt{v_1}} \approx 3.16 \thinspace \mathrm{sign}(g_1)$ → step **≈ 3× too big**.
- With correction: $\dfrac{\hat{m}_1}{\sqrt{\hat{v}_1}} = \mathrm{sign}(g_1)$ → **unit-size step**.

(Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015.)

#### Adam (full form)

```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations + 1):
    dtheta = compute_gradient(theta)
    first_moment  = beta1 * first_moment  + (1 - beta1) * dtheta            # momentum
    second_moment = beta2 * second_moment + (1 - beta2) * dtheta * dtheta   # RMSProp
    first_unbias  = first_moment  / (1 - beta1 ** t)                        # bias correction
    second_unbias = second_moment / (1 - beta2 ** t)
    theta -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)
```

$$
\theta_t = \theta_{t-1} - \eta \thinspace \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$

The bias correction compensates for $m$ and $v$ being initialized at zero and needing time to "warm up".

> ⭐ **Good default:** Adam with `beta1 = 0.9`, `beta2 = 0.999`, `learning_rate = 1e-3` or `5e-4` is a great starting point for many models.

### 3.9 AdamW: Decoupled Weight Decay

**Adam + L2 regularization ≠ weight decay.**

1. L2 adds $\frac{\lambda}{2} \lVert \theta \rVert^2$ to the loss, so $\lambda \theta$ joins the gradient:

$$
g_t = \nabla \mathcal{L}(\theta_{t-1}) + \lambda \theta_{t-1}
$$

2. With **SGD**, L2 and weight decay give the **same** update:

$$
\theta_t = \theta_{t-1} - \eta \nabla \mathcal{L}(\theta_{t-1}) - \eta \lambda \theta_{t-1}
$$

3. With **Adam**, $\lambda \theta$ passes through $\hat{m}$ and $\hat{v}$, so the decay term is divided by $\sqrt{\hat{v}}$ like the rest of the step:

$$
\text{decay of weight } i \approx \frac{\eta \lambda}{\sqrt{\hat{v}_{t, i}}} \thinspace \theta_{t-1, i}
$$

| Optimizer | Shrink rate of weight $i$ |
|---|---|
| SGD + L2 | $\eta \lambda$ — same for every weight |
| Adam + L2 | $\dfrac{\eta \lambda}{\sqrt{\hat{v}_i}}$ — weights with large past gradients barely shrink |

**AdamW** decouples the decay from the adaptive step:

$$
\theta_t = \theta_{t-1} - \eta \Big(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \theta_{t-1}\Big)
$$

```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations + 1):
    dtheta = compute_gradient(theta)          # loss gradient only, no lambda*theta
    first_moment  = beta1 * first_moment  + (1 - beta1) * dtheta
    second_moment = beta2 * second_moment + (1 - beta2) * dtheta * dtheta
    first_unbias  = first_moment  / (1 - beta1 ** t)
    second_unbias = second_moment / (1 - beta2 ** t)
    theta -= learning_rate * weight_decay * theta                            # weight decay
    theta -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7)  # Adam step
```

$\lambda \theta$ never enters $\hat{m}$ or $\hat{v}$, so **every weight shrinks at the same rate $\eta \lambda$**.

> PyTorch: `torch.optim.AdamW` (default `weight_decay = 0.01`).
> Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.

### 3.10 Optimizer Comparison

| Algorithm | Tracks 1st moments (momentum) | Tracks 2nd moments (adaptive LR) | Leaky 2nd moments | Bias correction |
|---|:---:|:---:|:---:|:---:|
| SGD | ✗ | ✗ | ✗ | ✗ |
| SGD + Momentum | ✓ | ✗ | ✗ | ✗ |
| Nesterov | ✓ | ✗ | ✗ | ✗ |
| AdaGrad | ✗ | ✓ | ✗ | ✗ |
| RMSProp | ✗ | ✓ | ✓ | ✗ |
| Adam | ✓ | ✓ | ✓ | ✓ |

---

## 4. Learning-Rate Schedules

**Decay the learning rate over the learning process**: a large LR makes fast progress early but bounces around the minimum; decaying it lets the iterates settle in. Watching the loss curve, you'll often see a sharp drop each time the LR is lowered ("decay learning rate manually").

Let $\alpha_0$ = initial LR, $t$ = current epoch, $T$ = total epochs.

| Schedule | Formula | Shape |
|---|---|---|
| **Step decay** | $\alpha$ multiplied by a constant (e.g. 0.1) at fixed epochs | staircase |
| **Linear decay** | $\alpha = \alpha_0 \big(1 - t / T\big)$ | straight line to 0 |
| **Cosine decay** | $\alpha = \dfrac{\alpha_0}{2}\big(1 + \cos(\pi t / T)\big)$ | smooth S-curve to 0 |
| **Exponential decay** | $\alpha = \alpha_0 \thinspace e^{-k t}$ | fast early drop, long tail |
| **Linear warm-up** | ramp $\alpha$ linearly from ≈0 to $\alpha_0$ over the first few epochs, then decay | spike then decay |

Warm-up pairs naturally with Adam: early on the moment estimates are unreliable, so small first steps are safer.

---

## 5. Hyperparameter Tuning

**Hyperparameters to play with**
- network architecture
- learning rate, its decay schedule, update type (optimizer)
- regularization (L2 / Dropout strength)

*(Practitioner as a DJ at a mixing board — the "music" is the loss function.)*

**Procedure**
1. Decide on a set of hyperparameters to test.
2. Train a model for each setting.
3. Compute the **validation** loss.
4. Keep the hyperparameters with the best validation loss.

**Grid search**: define a set of values for every hyperparameter and try all combinations. Search on a **logarithmic grid** (e.g. $\log(\text{learning rate})$ × $\log(\text{regularization})$).

**Random search**: sample hyperparameters at random (again on log scale).

**Random search > grid search.** If one hyperparameter (say regularization) barely matters, a $3 \times 3$ grid only tries **3 distinct** values of the important one (learning rate), while 9 random samples try **9 distinct** values of it.

```
Grid search                 Random search
 reg ▲                       reg ▲
     │ •   •   •                 │   •      •
     │ •   •   •                 │ •    •       •
     │ •   •   •                 │      •  •  •    •
     └───────────► lr            └───────────► lr
   only 3 distinct lr          9 distinct lr values
```

---

## 6. Cheat Sheet

| Topic | Key takeaway |
|---|---|
| Zero init | All gradients zero / symmetric → never learns |
| $\mathcal{N}(0, 1)$ init | $\mathbb{E}[y^2] = N \tau^2$ → explodes with depth |
| Xavier | $\sigma^2 = 2 / (N_{\text{in}} + N_{\text{out}})$ — good for linear / tanh |
| Kaiming | $\sigma^2 = 2 / N_{\text{in}}$ — good for ReLU; **PyTorch default** |
| Transfer learning | Reuse pretrained layers as a feature extractor; retrain the head (or fine-tune all) |
| SGD | Noisy, oscillates in ravines, slow in flat directions |
| Newton | Uses $\nabla^2 \mathcal{L}$; too costly ($d \times d$ matrix + inverse) |
| Momentum | Accumulates velocity → damps oscillation, accelerates along ravines |
| Nesterov | Gradient at look-ahead point; "accelerated GD" |
| AdaGrad | Per-coordinate LR $\propto 1 / \sqrt{\sum g^2}$; LR only decreases |
| RMSProp | AdaGrad with EWMA → forgets stale gradients |
| Adam | Momentum + RMSProp + bias correction; default $\beta_1 = 0.9$, $\beta_2 = 0.999$, LR $10^{-3}$ |
| AdamW | Weight decay applied outside the adaptive step; use instead of Adam + L2 |
| LR schedules | Step, linear, cosine, exponential, warm-up |
| Tuning | Use validation loss; log scale; random search beats grid search |

---

### References

- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010 (Xavier init)
- He et al., *Delving Deep into Rectifiers*, ICCV 2015 (Kaiming init)
- Sutskever et al., *On the importance of initialization and momentum in deep learning*, ICML 2013
- Bottou & Bousquet, *The Tradeoffs of Large Scale Learning*, NeurIPS 2007
- Tieleman & Hinton, *Lecture 6.5 — RMSProp*, COURSERA 2012
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019
- PyTorch Transfer Learning Tutorial — <https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html>
