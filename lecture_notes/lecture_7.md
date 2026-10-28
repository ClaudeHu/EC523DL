# Deep Learning: Training I — Minibatching and Regularization

> Lecture notes (EC523). Topics: tensors and contraction, minibatch SGD, regularization via noise (random, adversarial, augmentation), dropout / DropBlock, normalization (BatchNorm vs LayerNorm), NumPy broadcasting.

---

## 0. Logistics (from slides)

- Meet your project teammates and read the project PDF carefully.
- **Midway + Proposal due Nov 2.** You should have a ranking on the leaderboard by then. HW extensions do **not** apply to project deadlines.
- Short Quiz 2: Oct 28 (covers HW2–HW3).
- Do the PyTorch tutorial if you don't know PyTorch yet.

**Recap of last lecture: backpropagation**

- Backprop automatically computes derivatives of complicated functions at arbitrary points. It is a careful application of the chain rule.
- PyTorch / TensorFlow sped up ML research mainly through (1) easy autograd and (2) easy GPU support.

---

## 1. Matrices and Tensors

- A **matrix** in $\mathbb{R}^{n \times m}$ is a 2-D array: a "list of lists" (outer list size $n$, inner lists size $m$).
- A **tensor** in $\mathbb{R}^{d_1 \times \cdots \times d_n}$ is an $n$-dimensional array: a "list of lists of ... of lists".
- A tensor in $\mathbb{R}^{a \times b \times c}$ has **shape** `[a, b, c]`.
- Access via multi-indices, e.g. `A[0, 2, 3]`.

### 1.1 Tensor contraction (generalized matrix multiplication)

Two tensors $A$ and $B$ can be contracted along **one** dimension if the **last** entry of $A$'s shape equals the **first** entry of $B$'s shape.

- `A.shape = [2, 45, 7]`, `B.shape = [7, 3, 67]`
- Remove the shared entry and concatenate: `contract(A, B).shape = [2, 45, 3, 67]`

If $A$ has shape $[a_1, \dots, a_n, c]$ and $B$ has shape $[c, b_1, \dots, b_m]$:

```math
\operatorname{contract}(A, B)[a_1, \dots, a_n, b_1, \dots, b_m] = \sum_{i=1}^{c} A[a_1, \dots, a_n, i] \cdot B[i, b_1, \dots, b_m]
```

With $n = m = 1$ this is ordinary matrix multiplication:

```math
\operatorname{contract}(A, B)[a, b] = \sum_{i=1}^{c} A[a, i] \cdot B[i, b]
```

### 1.2 Contraction along many dimensions

Contracting along $k$ dimensions requires the **last $k$** dims of $A$ to match the **first $k$** dims of $B$.

- `A.shape = [2, 5, 4, 6]`, `B.shape = [4, 6, 8, 2]` → `contract(A, B, 2).shape = [2, 5, 8, 2]`
- This is `np.tensordot(A, B, axes=k)`.

If $A$ has shape $[a_1, \dots, a_n, c_1, \dots, c_k]$ and $B$ has shape $[c_1, \dots, c_k, b_1, \dots, b_m]$:

```math
\operatorname{contract}(A, B, k)[a_1, \dots, a_n, b_1, \dots, b_m] = \sum_{i_1=1}^{c_1} \cdots \sum_{i_k=1}^{c_k} A[a_1, \dots, a_n, i_1, \dots, i_k] \cdot B[i_1, \dots, i_k, b_1, \dots, b_m]
```

### 1.3 A minibatch as a tensor

- $X$ has shape `[B, m, n]` (a batch of $B$ matrices), $Y$ has shape `[m, n]`.
- `contract(X, Y, 2)` has shape `[B]`, with

```math
Z[i] = \sum_{j=1}^{m} \sum_{k=1}^{n} X[i, j, k] \cdot Y[j, k]
```

In NumPy: `Z[i] = np.sum(X[i] * Y)`.

### 1.4 Derivatives of tensor-valued functions (marked "skip" in lecture)

For $f : \mathbb{R}^{d_1 \times d_2 \times d_3} \to \mathbb{R}^{a \times b}$, the derivative is a tensor

```math
D[f](x) \in \mathbb{R}^{d_1 \times d_2 \times d_3 \times a \times b}, \qquad D[f](x)[i, j, k, z, w] = \frac{\partial f_{z,w}}{\partial x_{i,j,k}}(x)
```

It contracts all input dims and produces the output dims. For scalar $f : \mathbb{R}^{d_1 \times d_2 \times d_3} \to \mathbb{R}$, drop the trailing $\times 1$ so $D[f](x) \in \mathbb{R}^{d_1 \times d_2 \times d_3}$, and

```math
\operatorname{contract}(X, D[f](x)) = \operatorname{sum}\left(X \odot D[f](x)\right)
```

where $\odot$ is element-wise multiplication (`*` in NumPy).

---

## 2. Minibatching

### 2.1 SGD recap

```math
\theta_{t+1} = \theta_t - \eta \nabla \ell(\theta_t, x_t)
```

- $\theta$: network parameters
- $x_t$: the $t$-th training example
- $\eta \in \mathbb{R}$: learning rate

Why it works: the single-example gradient is an **unbiased** estimate of the full gradient,

```math
\mathbb{E}\left[\nabla \ell(\theta_t, x_t)\right] = \nabla \mathcal{L}(\theta_t)
```

where $\mathcal{L}$ is the average loss we want to minimize. **Problem:** $\nabla \ell(\theta_t, x_t)$ has high **variance**, which slows SGD down.

### 2.2 Minibatch SGD

Sample a minibatch $(x_1, \dots, x_B)$ and average the gradients:

```math
\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i=1}^{B} \nabla \ell(\theta_t, x_i)
```

The average is a **better estimate** of $\nabla \mathcal{L}(\theta_t)$. If the per-example gradients are i.i.d. with variance $\sigma^2$, the minibatch average has the same mean but variance

```math
\operatorname{Var}\left[\frac{1}{B} \sum_{i=1}^{B} \nabla \ell(\theta_t, x_i)\right] = \frac{\sigma^2}{B}
```

so the noise (standard deviation) shrinks like $1/\sqrt{B}$.

> **Example (F1 race pace):** One lap time is a noisy estimate of a car's true race pace — traffic, a lock-up, a gust of wind. Averaging a 10-lap stint gives a much tighter estimate. Going from 1 lap to 4 laps halves the noise ($1/\sqrt{4}$); going from 4 to 16 halves it again. Same idea: more examples per step → steadier gradient, with diminishing returns.

### 2.3 Data parallelism

Sums and averages are easy to parallelize. Split the batch across, e.g., 3 workers (GPUs):

```math
\frac{1}{B}\sum_{i=1}^{B} \nabla \ell(\theta_t, x_i) = \underbrace{\frac{1}{B}\sum_{i=1}^{B/3} \nabla \ell(\theta_t, x_i)}_{\text{GPU 1}} + \underbrace{\frac{1}{B}\sum_{i=B/3+1}^{2B/3} \nabla \ell(\theta_t, x_i)}_{\text{GPU 2}} + \underbrace{\frac{1}{B}\sum_{i=2B/3+1}^{B} \nabla \ell(\theta_t, x_i)}_{\text{GPU 3}}
```

Each worker computes its partial sum independently; the results are added.

---

## 3. Regularization via Noise

- **Previously:** prevent overfitting by "throwing out" some parameter values (restricting the model).
- **Now:** prevent overfitting by asking the model to fit **noise in addition to** the training data.

Intuition:

- A neural net is very expressive and can fit almost anything.
- Adding noise makes the fitting problem much **harder**.
- The network may "give up" on some data — hopefully the data it gives up on is noise or outliers.

### 3.1 Adversarial noise

Replace $x$ with $x + \delta$, where $\delta$ is the **worst** small perturbation:

```math
\delta = \arg\max_{\lVert \delta \rVert \le r} \ell(\theta_t, x + \delta)
```

- $\lVert \delta \rVert \le r$ keeps the perturbation small.
- Adversarial noise is a **targeted** version of random noise: with enough random noise you would eventually hit the adversarial direction.
- It is harder to compute than random noise.

**Computing $\delta$:** run a few steps of gradient optimization on $\delta$ (gradient **ascent** on the loss, since we maximize), projecting back onto the ball $\lVert \delta \rVert \le r$ after each step:

```math
\delta \leftarrow \Pi_{\lVert \delta \rVert \le r}\left(\delta + \alpha \nabla_{\delta}\, \ell(\theta_t, x + \delta)\right)
```

(Standard names, beyond the slides: one step with the sign of the gradient is **FGSM**; several projected steps is **PGD**.)

### 3.2 Adversarial examples

- Typical image classifiers are **not robust** to adversarial examples.
  - Szegedy et al., *Intriguing properties of neural networks*: imperceptible perturbations make a school bus, a dog, and a temple all get classified as "ostrich".
  - Brown et al., *Adversarial Patch*: a printed sticker placed next to a banana makes the classifier say "toaster".
- **Adversarial training** (training on $x + \delta$) improves robustness **and** helps with overfitting.
- Downside: it is slow, since every step needs an inner optimization to generate $\delta$.

### 3.3 Dataset augmentation

- Noise regularization assumes **similar inputs → similar outputs**: a small perturbation shouldn't change the answer.
- Sometimes we have **domain knowledge** that even quite different inputs should have the same (or a known) output.
- For images, apply **geometric** and **photometric** transformations:
  - random crops
  - horizontal flips
  - random rotations, scales, and crops
  - random photometric shifts (brightness, contrast, color, grayscale)

> **Key rule:** an augmentation must **preserve the label**. A flipped cat is still a cat, but a flipped "b" becomes a "d".
>
> **Example (classical guitar recordings):** Training a model to classify playing technique (rest stroke, tremolo, rasgueado) from audio? Safe augmentations: small gain changes, adding room reverb, mild background noise, swapping microphone EQ curves — a tremolo is still a tremolo. But **pitch-shifting** would be a bad augmentation for a *note transcription* model, because it changes the label (an E becomes an F).

---

## 4. Dropout

Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* (JMLR 2014).

### 4.1 Training time

**Ignore each hidden unit with some probability.** For each training example, sample a random mask: with keep probability $p$,

```math
m_j \sim \operatorname{Bernoulli}(p), \qquad \tilde{h}_j = m_j \cdot h_j
```

Dropped units output 0 and receive no gradient for that step.

### 4.2 Test time

**Use all activations, but scale them by the keep probability** so the expected input to the next layer matches training:

```math
h_j^{\text{test}} = p \cdot h_j
```

> **Implementation note (beyond the slides):** PyTorch's `nn.Dropout(p)` uses **inverted dropout**: here `p` is the **drop** probability, and the surviving activations are scaled by $1/(1-p)$ at **training** time, so nothing needs rescaling at test time. Remember `model.train()` vs `model.eval()`.

### 4.3 Why it works: an implicit ensemble

- With $N$ droppable units, there are $2^N$ possible sub-networks.
- **Dropout trains an ensemble of all $2^N$ sub-networks with shared weights.**
- The test-time scaling **approximates the geometric mean** of the ensemble's predictions.

> **Example (soccer):** A coach who randomly benches different starters in every training match forces the squad to stop routing every attack through the star striker. Every player must be able to contribute, so the team keeps working when someone is injured. Test time is the final: everyone plays, and the team performs like the average of all those lineups.

### 4.4 DropBlock (structured dropout for CNNs)

Ghiasi, Lin, Le — *DropBlock: A regularization method for convolutional networks*.

- In conv layers, neighboring activations are **spatially correlated**. Dropping random single units barely removes information, because nearby units carry the same information.
- **DropBlock** drops **contiguous blocks** of a feature map (e.g., the whole "head" region of a dog), which forces the network to use other evidence (body, feet).
- The paper reports ResNet-50 on ImageNet improving from 76.51% to 78.13%.

---

## 5. Normalization

### 5.1 Motivation: units matter

Linear model: $\hat{y} = w \cdot x$, where $x$ is weather measurements and $\hat{y}$ is predicted rainfall.

- If windspeed is in **light-years per second** (absurdly tiny) and pressure is in **picopascals** (absurdly huge), the optimal weights are skewed in the opposite directions.
- Gradients for windspeed become tiny while gradients for pressure become huge → the weights are very hard to learn with one learning rate.

> **Example (F1 telemetry):** Features like engine RPM (~12,000), speed (~300 km/h), and lateral g (~5) live on wildly different scales. Without normalization, the RPM weight dominates the gradient and the g-force weight barely moves.

### 5.2 Standardization

```math
x \mapsto \frac{x - \mathbb{E}[x]}{\sqrt{\operatorname{Var}(x)}}
```

The result satisfies $\mathbb{E}[x] = 0$ and $\operatorname{Var}(x) = 1$.

### 5.3 Batch Normalization

For each neuron (feature), estimate the statistics **from the current minibatch**:

```math
\mu = \frac{1}{B} \sum_{i=1}^{B} x_i
```

```math
\sigma^2 = \frac{1}{B} \sum_{i=1}^{B} (x_i - \mu)^2
```

```math
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

($\epsilon$ is a small constant for numerical stability, used in practice.)

**Learnable shift and scale.** Why force mean 0 and variance 1? Let the network choose a mean $\beta$ and variance $\gamma^2$:

```math
y_i = \gamma \hat{x}_i + \beta
```

How to choose $\gamma$ and $\beta$? **Just learn them** (by backprop, like any other parameter).

Placement: a BatchNorm layer (with its own $\gamma$ and $\beta$) is inserted between blocks, e.g. `x → linear → BatchNorm → ReLU → ... → y`.

**Worked example.** One neuron sees the batch $[5, 1, 4]$:

```math
\mu = \frac{5 + 1 + 4}{3} = 3.33, \qquad \sigma^2 = \frac{1.67^2 + 2.33^2 + 0.67^2}{3} = 2.89, \qquad \sigma = 1.70
```

```math
\hat{x} = \left[\frac{5 - 3.33}{1.70}, \frac{1 - 3.33}{1.70}, \frac{4 - 3.33}{1.70}\right] = [0.98, -1.37, 0.39]
```

### 5.4 BatchNorm at inference

- BatchNorm only makes sense with batch size **much larger than 1**. With batch size 1, $x_i - \mu = 0$, so every output is just $\beta$.
- At inference, batch size is often 1.
- Fix: after training, compute fixed $\mathbb{E}[x]$ and $\operatorname{Var}(x)$ by averaging over the training set (PyTorch keeps running averages during training), and use those constants at inference.

### 5.5 Batch norm in action (lecture demo)

- Task: fit a 1-D target curve with a 1-hidden-layer, 100-unit ReLU network.
- **Without BatchNorm** (plain init): the fit is poor — mostly straight-line segments that miss the steps. Last lecture this was fixed with a manual initialization trick (`self.Linear1.bias *= 80`).
- **With `nn.BatchNorm1d(num_features=1)` applied to the input** (`out = self.batchnorm(x)` before `Linear1`): the fit matches the target almost exactly.

Takeaways:

- BatchNorm usually does **not** increase the network's capacity: at inference it is a linear map.
- Instead, it can make **training much faster**.
- The "units" argument is **not rigorous** — it is genuinely unclear why BatchNorm works.
  - Ioffe and Szegedy (original paper) claimed it reduces "internal covariate shift".
  - Santurkar et al., *How Does Batch Normalization Help Optimization?*, argue the distributional stability of layer inputs has little to do with BatchNorm's success.

### 5.6 Layer Normalization

Problems with BatchNorm:

- Hard to use without large batches.
- Effectiveness varies by architecture.
- It may **discard relationships between features.** Example: if one feature is the temperature at 1pm and another is the temperature at 1am, the 1pm value is probably higher. BatchNorm normalizes each feature independently and loses this ordering.

**BatchNorm** normalizes each neuron **independently across the batch**.
**LayerNorm** normalizes each example **across all neurons in the layer**.

For one example with layer activations $x[1], \dots, x[H]$ ($H$ = layer size):

```math
\mathbb{E}[x] = \frac{1}{H} \sum_{i=1}^{H} x[i]
```

```math
\operatorname{Var}(x) = \frac{1}{H} \sum_{i=1}^{H} \left(x[i] - \mathbb{E}[x]\right)^2
```

```math
x[i] \leftarrow \gamma[i] \frac{x[i] - \mathbb{E}[x]}{\sqrt{\operatorname{Var}(x) + \epsilon}} + \beta[i]
```

**Worked example.** One example's layer activations are $[5, 4, 7, 2]$:

```math
\mathbb{E}[x] = 4.5, \qquad \operatorname{Var}(x) = \frac{0.25 + 0.25 + 6.25 + 6.25}{4} = 3.25, \qquad \sqrt{3.25} = 1.80
```

```math
\hat{x} = [0.28, -0.28, 1.39, -1.39]
```

> **Mental picture:** arrange activations as a matrix with rows = neurons and columns = examples in the batch. **BatchNorm normalizes each row; LayerNorm normalizes each column.**

### 5.7 BatchNorm vs LayerNorm

| | BatchNorm | LayerNorm |
|---|---|---|
| Statistics computed over | the batch (per neuron) | the layer (per example) |
| Requires | batch size > 1 | layer size > 1 |
| Inference | needs fixed mean/variance from the training set | no changes needed |
| At inference it is | **linear** | **non-linear** |
| Changes network capacity? | usually no | yes |

**Why BatchNorm is linear at inference:**

```math
x[i] \leftarrow \gamma[i] \frac{x[i] - \mathbb{E}[x[i]]}{\sqrt{\operatorname{Var}(x[i])}} + \beta[i]
```

Here $\mathbb{E}[x[i]]$ and $\operatorname{Var}(x[i])$ are constants from the training set, so this is just a specific (diagonal) linear layer. A network with BatchNorm layers can represent the **same functions** as one without them.

**Why LayerNorm is non-linear:** $\mathbb{E}[x]$ and $\operatorname{Var}(x)$ depend on the current input, and $\operatorname{Var}(x)$ depends on it non-linearly. So a network with LayerNorm layers does **not** represent the same function class as one without them.

---

## 6. Extra (DIY): NumPy Broadcasting

Adding an array of shape `[2, 3, 4]` to one of shape `[1, 4]`: the `[1, 4]` array is **broadcast** to `[2, 3, 4]`. To compute `A + B` with different shapes, NumPy tries to broadcast `B` to `A`'s shape and returns the sum.

Broadcasting "promotes" smaller tensors to bigger ones. Two rules:

1. **New dimensions are added to the front** of the shape.
   - `[22, 3]` → `[4, 22, 3]` ✅  `[22, 3]` → `[22, 3, 4]` ❌
2. **A dimension can change size only if its original size was 1.**
   - `[22, 1, 3]` → `[22, 4, 3]` ✅  `[22, 1, 3]` → `[22, 1, 4]` ❌

**Indexing a broadcast array:** drop the added leading dimensions, and use index 0 for every dimension that was originally size 1.

- `A.shape = [2, 3, 1, 4]`
- `B.shape = [100, 23, 2, 3, 99, 4]` is a broadcast of `A`.
- `B[33, 22, 1, 2, 60, 3] = A[1, 2, 0, 3]`
  - drop `33, 22` (new front dims) → `[1, 2, 60, 3]`
  - `60` sits in a size-1 dim of `A` → becomes `0`

> ⚠️ The slide writes `A[1, 2, 0, 4]`, but the last dimension of `A` has size 4 (valid indices 0–3), so the correct index is `A[1, 2, 0, 3]`.

---

## Quick Review Checklist

- [ ] Can I compute the output shape of `np.tensordot(A, B, axes=k)`?
- [ ] Why does minibatching help SGD? (variance scales as $\sigma^2 / B$)
- [ ] How is adversarial noise computed, and why is adversarial training slow?
- [ ] What makes an augmentation valid? (it must preserve the label)
- [ ] How does dropout differ at train vs test time? What does "ensemble of $2^N$ sub-networks" mean?
- [ ] Why is plain dropout weak for CNNs, and what does DropBlock do instead?
- [ ] Why does BatchNorm fail with batch size 1, and how is inference handled?
- [ ] BatchNorm vs LayerNorm: which axis, which is linear at inference, which changes capacity?
- [ ] Apply the two broadcasting rules to index a broadcast array.
