# EC523 Quiz Review: HW2 & HW3

> A 15-minute written quiz probably means **short derivations, shape and parameter counting, definitions, and "what happens if…" questions**. Each section has the key formulas, then the worked answers from the homework, then the traps.

---

## Part A: HW2

### A1. Universal approximation with ReLU

**Setup:** one hidden layer, linear plus bias, ReLU, then linear with no bias:

$$
\hat{y} = \sum_{i} a_i \max(0, b_i x + c_i)
$$

**Key idea:** the difference of two shifted ReLUs gives a **ramp that saturates**. Stacking ramps gives bumps, and summing bumps gives any function.

#### Q1.1: Step function $\mathbf{1}[x>0]$ with 2 units

$$
\hat{y}(x) = \frac{1}{\delta}\text{ReLU}(x) - \frac{1}{\delta}\text{ReLU}(x-\delta)
$$

So $a_1 = \frac{1}{\delta}$, $b_1 = 1$, $c_1 = 0$, and $a_2 = -\frac{1}{\delta}$, $b_2 = 1$, $c_2 = -\delta$.

| Region | Value |
|---|---|
| $x \le 0$ | $0$ |
| $0 < x < \delta$ | $x/\delta$ (linear ramp) |
| $x \ge \delta$ | $\frac{x}{\delta} - \frac{x-\delta}{\delta} = 1$ |

#### Q1.2: Impulse $u_\delta(x) = \mathbf{1}[0 \le x \le \delta]$ with 4 units

The target shape is 0 before $-\delta/4$, rising to 1 at $\delta/4$, flat at 1 until $3\delta/4$, falling to 0 at $5\delta/4$.

- Each ramp spans a width of $\delta/2$, so the slope is $\frac{1}{\delta/2} = \frac{2}{\delta}$.
- The result is an up-step minus a later up-step:

$$
\hat{u}_\delta(x) = \frac{2}{\delta}\Big[\text{ReLU}\big(x+\tfrac{\delta}{4}\big) - \text{ReLU}\big(x-\tfrac{\delta}{4}\big) - \text{ReLU}\big(x-\tfrac{3\delta}{4}\big) + \text{ReLU}\big(x-\tfrac{5\delta}{4}\big)\Big]
$$

The slopes add up to $+\frac{2}{\delta}$, then $0$, then $-\frac{2}{\delta}$, then $0$. Check that the signs alternate **+ − − +**.

```python
def hat_u_delta(x, delta):
    k = 2 / delta
    return k * (relu(x + delta/4) - relu(x - delta/4)
              - relu(x - 3*delta/4) + relu(x - 5*delta/4))
```

#### Q1.4: Riemann-sum approximation

$$
\hat{f}(x) = \sum_{i=0}^{N-1} f(a + i\delta) \cdot u_\delta(x - a - i\delta), \qquad \delta = \frac{b-a}{N}
$$

- Larger $N$ means a narrower $\delta$ and a better fit. Each impulse costs 4 hidden units, so the network uses $4N$ units.
- **Trap:** the notebook writes $\lfloor (b-a)/N \rfloor$. With $b-a = 2\pi$ and $N = 10$, the floor gives 0. Use plain division.

```python
def hat_f(x, N, a, b):
    delta = (b - a) / N
    return sum(f(a + i*delta) * hat_u_delta(x - a - i*delta, delta) for i in range(N))
```

#### Q1.5: Why a polynomial activation is not universal (single hidden layer)

With $\sigma(t) = t^2 + t + 1$:

$$
\hat{y} = \sum_i a_i \left[(b_i x + c_i)^2 + (b_i x + c_i) + 1\right] = \alpha x^2 + \beta x + \gamma
$$

- **No matter how many hidden units**, the output is always a polynomial of degree at most 2.
- **Counterexamples** it cannot fit well: $\sin(x)$ on $[0, 2\pi]$ (multiple wiggles), $x^3$, the step function, $e^x$ on a wide interval, or $|x|$ beyond a crude parabola.
- General fact: a single-hidden-layer network is universal **if and only if** the activation is **not a polynomial** (Leshno et al.).

#### Q1.6 (Bonus): Depth makes the quadratic activation universal

These two identities are exact:

$$
\sigma(t) - \sigma(-t) = 2t \quad \text{(identity, so values can pass through layers)}
$$

$$
\sigma(t) + \sigma(-t) - 2 = 2t^2 \quad \text{(squaring)}
$$

The argument has three steps:

1. **Products:** $xy = \frac{1}{4}\left[(x+y)^2 - (x-y)^2\right]$, so a layer can multiply two values.
2. **Polynomials:** with depth, repeated products and sums build **any polynomial**. Composing $L$ layers gives degree up to $2^L$.
3. **Stone–Weierstrass:** polynomials are dense in continuous functions on a compact set. Therefore deep quadratic networks are universal approximators.

---

### A2. Autograd: forward and backward rules

**Backprop rule:** each layer receives the upstream gradient $G = \frac{\partial L}{\partial \text{out}}$ and does two things:

- returns $\frac{\partial L}{\partial \text{in}} = \left(\frac{\partial\, \text{out}}{\partial\, \text{in}}\right)^{T} G$, which is passed to the previous layer;
- stores $\frac{\partial L}{\partial \text{param}}$ in `self.grad`.

The loss layer starts the chain with `backward(1)`.

#### Multiplication layers (Q2.1)

| Layer | Forward | $\partial L / \partial \text{input}$ | $\partial L / \partial \text{param}$ |
|---|---|---|---|
| Linear (math form) | $Z = WX$ | $W^{T} G$ | $G X^{T}$ |
| Linear (code form, batch $B \times N$) | $Z = XW$ | $G W^{T}$ | $X^{T} G$ |
| Bias | $Z = X + b$ | $G$ (identity) | $\sum_{\text{batch}} G$ |

- **Shape check:** if $X$ is $B \times N$ and $W$ is $N \times M$, then $G$ is $B \times M$, $X^{T} G$ is $N \times M$ (same as $W$), and $G W^{T}$ is $B \times N$ (same as $X$).
- $\frac{\partial Z_{\text{bias}}}{\partial X} = I$.

```python
class Linear(Layer):
    def forward(self, input):
        self.input = input
        return input @ self.parameter
    def backward(self, downstream_grad):
        self.grad = self.input.T @ downstream_grad
        return downstream_grad @ self.parameter.T
```

#### Activation layers (Q2.2)

**ReLU**

$$
\frac{d}{dx}\text{ReLU}(x) = \mathbf{1}[x>0]
$$

The backward pass is `G * (x > 0)`.

**Sigmoid**

$$
\sigma'(x) = \sigma(x)\left(1-\sigma(x)\right)
$$

- Save the forward **output** and reuse it in the backward pass.
- The maximum value is $0.25$ at $x = 0$, which causes **vanishing gradients** for large $|x|$.

**GELU (tanh approximation)**

Let $z = \sqrt{2/\pi} \left(x + 0.044715 x^3\right)$, so that $\text{GELU}(x) \approx \frac{x}{2}\left(1+\tanh z\right)$.

By the product rule plus the chain rule:

$$
\frac{d}{dx}\text{GELU}(x) \approx \frac{1}{2}\left(1+\tanh z\right) + \frac{x}{2}\left(1-\tanh^2 z\right)\sqrt{\frac{2}{\pi}}\left(1 + 3 \cdot 0.044715 x^2\right)
$$

- This uses $\frac{d}{dz}\tanh z = 1 - \tanh^2 z = \text{sech}^2 z$; the homework wants the answer written in terms of $\tanh$.
- GELU is smooth and slightly negative for $x < 0$, so it has no "dead neurons" the way ReLU does.

#### Loss layer (Q2.3)

$$
\text{MSE} = \frac{1}{2N}\sum_{i=1}^{N}\left(y_i - \hat{y}_i\right)^2 \quad \Longrightarrow \quad \frac{\partial\, \text{MSE}}{\partial \hat{y}_i} = \frac{\hat{y}_i - y_i}{N}
$$

The $\frac{1}{2}$ exists to cancel the 2 from differentiating the square.

```python
def forward(self, input):
    self.input = input
    return np.sum((input - self.labels)**2) / (2 * input.shape[0])
def backward(self, downstream_grad):
    return downstream_grad * (self.input - self.labels) / self.input.shape[0]
```

#### Gradient check (Q2.4)

| Method | Formula | Error order |
|---|---|---|
| Forward difference (used in Q2) | $\frac{f(x+h) - f(x)}{h}$ | $O(h)$ |
| Centered difference (used in Q3) | $\frac{f(x+h) - f(x-h)}{2h}$ | $O(h^2)$, more accurate |

The check compares the relative error $\frac{\lVert g_{\text{analytic}} - g_{\text{numeric}} \rVert}{\lVert g_{\text{numeric}} \rVert}$ against a small threshold (for example $10^{-5}$).

---

### A3. Two-layer MLP (Q3)

- **Architecture:** Linear → Bias → Sigmoid → Linear → Bias, trained with MSE.
- **Initialization:** $W \sim \mathcal{N}(0, \text{std}^2)$ and $b = 0$.

```python
self.params['W1'] = std * np.random.randn(input_size, hidden_size)
self.params['W2'] = std * np.random.randn(hidden_size, label_size)
self.params['b1'] = np.zeros(hidden_size)
self.params['b2'] = np.zeros(label_size)
```

**Inside `loss()`:**

```python
scores = forward_layers(self.models, X)
loss_layer = MeanSquaredError(y)
loss = loss_layer.forward(scores)
```

**SGD update:** $\theta \leftarrow \theta - \eta \nabla_\theta L$.

**Trap:** the custom layers **never zero their gradients**. That is why re-running the training cell changes the curve, and why you should use "Run all" before submitting.

---

### A4. PyTorch basics (Q4)

#### Q4.1: Computation graph for $f = x + \sqrt{z} \cdot y$ with $x = -2$, $y = 5$, $z = 4$

**Forward pass:**

- $g = \sqrt{z} = 2$
- $q = g \cdot y = 10$
- $f = x + q = 8$

**Backward pass:**

| Gradient | Formula | Value |
|---|---|---|
| $\frac{\partial f}{\partial x}$ | $1$ | **1** |
| $\frac{\partial f}{\partial y}$ | $\sqrt{z}$ | **2** |
| $\frac{\partial f}{\partial z}$ | $\frac{y}{2\sqrt{z}}$ | **1.25** |

**Facts to remember:**

- `q.grad` is `None`. Intermediate (non-leaf) tensors do not keep gradients unless you call `q.retain_grad()` or use a hook.
- Printed tensors show `grad_fn=<SqrtBackward0>`, `<MulBackward0>`, and `<AddBackward0>`. Leaf tensors show `requires_grad=True` instead.
- `retain_graph=True` keeps the graph alive so you can call `backward()` again.
- **Gradients accumulate.** Call `x.grad.zero_()` (or `optimizer.zero_grad()`) before each backward pass.

#### Q4.2: Single neuron $g = \sigma(\mathbf{w}^{T}\mathbf{x})$

Use $\mathbf{x} = [-1, -2, 1]$ and $\mathbf{w} = [2, -3, -3]$, where the last entry acts as the bias.

**Forward pass:**

$$
z = (2)(-1) + (-3)(-2) + (-3)(1) = 1, \qquad g = \sigma(1) \approx 0.7311
$$

**Chain rule:**

$$
\frac{\partial g}{\partial z} = g(1-g) \approx 0.1966, \qquad \frac{\partial g}{\partial \mathbf{w}} = g(1-g) \cdot \mathbf{x}, \qquad \frac{\partial g}{\partial \mathbf{x}} = g(1-g) \cdot \mathbf{w}
$$

| Gradient | Value |
|---|---|
| $\partial g / \partial \mathbf{w}$ | $\approx [-0.197,\ -0.393,\ 0.197]$ |
| $\partial g / \partial \mathbf{x}$ | $\approx [0.393,\ -0.590,\ -0.590]$ |

#### Q4.3 and Q4.4: `nn.Linear` plus `nn.Module`

```python
class Neuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[2., -3.]]))
            self.fc.bias.copy_(torch.tensor([-3.]))
    def forward(self, x):
        return torch.sigmoid(self.fc(x))
```

- `fc.weight` has shape `(out, in)`. It is stored transposed relative to the math $\mathbf{w}^{T}\mathbf{x}$.
- **Trap:** for the result to match Q4.2, the input must be $[-1, -2]$. The notebook text says $[-1, 2]$, which gives $z = -11$ and $g \approx 1.7 \times 10^{-5}$.

#### Q4.5: Loss and SGD

- **`nn.CrossEntropyLoss`** = LogSoftmax + NLLLoss.
  - It expects **raw logits** of shape `(B, C)` and integer labels of shape `(B,)`.
  - It already applies softmax, so do **not** apply sigmoid or softmax first.
- **Quirk:** with a single output unit and labels in $\{0, 1\}$, cross-entropy does not work properly. Binary problems want either `nn.Linear(2, 2)` with cross-entropy, or one unit with `BCEWithLogitsLoss`.
- **`grad_fn` chain:** start at `loss.grad_fn` and follow `.next_functions[0][0]`. You will see roughly `NllLossBackward` → `LogSoftmaxBackward` → `SigmoidBackward` → `AddmmBackward` → `AccumulateGrad`.

**Standard training loop:**

```python
for i in range(n_iter):
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
```

#### Q4.6: Hidden-size ablation on SVHN (3 classes, 32×32 grayscale = 1024 inputs)

$$
\#\text{params} = (1024 \cdot h + h) + (h \cdot 3 + 3)
$$

| Hidden size $h$ | Parameters |
|---|---|
| 10 | 10,283 |
| 100 | 102,803 |

- **Expected result:** the larger hidden size reaches a lower training loss faster because it has more capacity. It also runs a higher risk of overfitting, and each step is slower.
- **Optimizer:** `optim.SGD(params, lr=0.001, momentum=0.9)`.
- **Device:** move both the model and every batch with `.to(device)`.

---

## Part B: HW3

### B1. SCC (Shared Computing Cluster) essentials

| Task | Command |
|---|---|
| Log in | `ssh user@scc1.bu.edu` |
| Check home-directory quota (10 GB) | `quota -s` |
| Check project quota | `pquota -u dl523` |
| List available GPUs | `qgpus` (or `qgpus -v`) |
| See GPU usage in a session | `nvidia-smi` |
| Submit a batch job | `qsub job.sh` |
| Check job status | `qstat -u user` |
| Delete a job | `qdel <job_id>` |

**Batch script template:**

```bash
#!/bin/bash -l
#$ -P dl523
#$ -l h_rt=01:00:00
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -N test_job
#$ -o output.txt
#$ -j y
module load python3 pytorch
python test_script.py
```

**Storage rule:** work in `/projectnb/dl523/students/<user>`, not in your home directory.

**Ranking GPUs:** compare memory size, compute capability, and architecture generation (e.g., A100 > V100 > P100), and consider how often each one is free on the queue.

---

### B2. CNN fundamentals

#### Output size of a convolution or pooling layer

$$
W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} - K + 2P}{S} \right\rfloor + 1
$$

**Useful shortcut:** with $K = 3$, $P = 1$, $S = 1$, the spatial size is preserved ("same" padding).

#### Parameter counts

$$
\text{Conv: } C_{\text{in}} \cdot C_{\text{out}} \cdot K^2 + C_{\text{out}}, \qquad \text{Linear: } n_{\text{in}} \cdot n_{\text{out}} + n_{\text{out}}
$$

#### Shape walk-through for the HW3 `Net` (input $3 \times 64 \times 64$)

| Layer | Output shape | Parameters |
|---|---|---|
| conv1 (3→16, k3, p1) | $16 \times 64 \times 64$ | $3 \cdot 16 \cdot 9 + 16 = 448$ |
| ReLU, MaxPool(2, 2) | $16 \times 32 \times 32$ | 0 |
| conv2 (16→32, k3, p1) | $32 \times 32 \times 32$ | $16 \cdot 32 \cdot 9 + 32 = 4640$ |
| ReLU, MaxPool(2, 2) | $32 \times 16 \times 16$ | 0 |
| flatten | $8192$ | 0 |
| fc1 | 128 | $8192 \cdot 128 + 128 = 1{,}048{,}704$ |
| fc2 | 64 | $8256$ |
| fc3 | 6 classes | $390$ |

The fully connected layers dominate the parameter count: fc1 alone is about 99% of the model.

#### Skip connection (the "code required" part)

- **The problem:** `residual` has shape $3 \times 64 \times 64$, but after conv2 the tensor has shape $32 \times 32 \times 32$. They cannot be added directly.
- **The fix:** project the residual with a 1×1 convolution with stride 2, which matches both channels and spatial size.

```python
self.skip = nn.Conv2d(3, 32, kernel_size=1, stride=2)   # in __init__
x = x + self.skip(residual)                              # after conv2 (and its norm layer)
```

**Why skip connections help:** gradients flow straight through the addition, $\frac{\partial (F(x) + x)}{\partial x} = F'(x) + 1$. This eases vanishing gradients and is the core idea of ResNet.

**Trap:** `Net` has **no ReLU between fc1, fc2, and fc3**. Linear followed by linear is still linear, so those three layers act as one linear map.

#### Q2.2 definitions (likely quiz material)

| Term | Definition | Setting in our net | Alternatives |
|---|---|---|---|
| **Stride** | Step size of the kernel as it slides | conv: 1, pool: 2 | 2 (downsampling conv) |
| **Padding** | Border added around the input to control output size | 1 (zeros) | 0 ("valid"), reflect, replicate |
| **Non-linearity** | Elementwise function that breaks linearity | ReLU | LeakyReLU, GELU, tanh, sigmoid, ELU |
| **Pooling** | Downsamples feature maps, giving some translation invariance | MaxPool 2×2 | AvgPool, global average pool, strided conv |
| **Loss** | Objective to minimize | CrossEntropy | NLL, focal loss, label smoothing |
| **Optimizer** | Rule for updating weights from gradients | SGD | Adam, AdamW, RMSProp |
| **Learning rate** | Step size $\eta$ | 0.005 | schedules: step, cosine, warmup |
| **Momentum** | Running average of past gradients that smooths updates | 0.9 | Nesterov momentum, or Adam's moments |

**Momentum update:**

$$
v_{t+1} = \mu v_t + \nabla L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

*Analogy:* momentum is like a race car's inertia through a chicane. It does not react to every tiny bump in the gradient, so it carries speed through flat regions and dampens zig-zagging in narrow valleys.

**Standard test loop:**

```python
def test_on_Pokemon(net, testloader):
    net.eval(); correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            pred = net(images).argmax(dim=1)
            total += labels.size(0); correct += (pred == labels).sum().item()
    return 100 * correct / total
```

Inside the training loop, the body is the same five lines as before: `zero_grad`, forward, compute loss, `backward`, `step`. Remember to move each batch with `.to(device)`.

---

### B3. Normalization

#### Input normalization

`Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))` maps pixel values from $[0, 1]$ to $[-1, 1]$:

$$
x' = \frac{x - 0.5}{0.5}
$$

Zero-centered inputs make the loss surface better conditioned, which usually gives faster and more stable training and higher accuracy.

#### BatchNorm vs LayerNorm

Both normalize and then apply a learnable scale $\gamma$ and shift $\beta$:

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta
$$

| Property | BatchNorm2d | LayerNorm |
|---|---|---|
| Statistics computed over | batch, H, W (separately per channel) | C, H, W (separately per sample) |
| PyTorch call | `nn.BatchNorm2d(16)` | `nn.LayerNorm([16, 64, 64])` (shape after conv1) |
| Depends on batch size? | Yes. Noisy with batch = 4 | No |
| Train vs eval behavior | Uses running mean and variance in `eval()` | Identical in both modes |
| Typical use | CNNs with large batches | Transformers, RNNs, small batches |

In the HW3 net, conv2's output is $32 \times 32 \times 32$, so the second LayerNorm is `nn.LayerNorm([32, 32, 32])`.

**Typical observations:**

- BatchNorm speeds up and stabilizes training and partly compensates for missing input normalization.
- With a tiny batch (4), BatchNorm statistics are noisy, so LayerNorm can be competitive.
- **Order:** conv → norm → ReLU.

#### Tuning knobs (Q2.4)

| Knob | Effect |
|---|---|
| More filters | More capacity, slower |
| Bigger kernel | Larger receptive field, more parameters |
| Pool stride | Changes the flattened size, so fc1's input must change too |
| Larger FC size | More parameters, more overfitting risk |
| Dropout | Regularization |
| Data augmentation | Flips and crops improve generalization |
| Early stopping | Prevents overfitting |

**Experiment rule:** change **one** variable at a time, keeping epochs and all other settings fixed.

---

### B4. Adversarial attack (Bonus): FGSM

$$
x' = x + \epsilon \cdot \text{sign}\left(\nabla_x J(\theta, x, y)\right)
$$

- The gradient is taken with respect to the **input** (`images.requires_grad_(True)`), not the weights.
- This is gradient **ascent** on the loss, with each pixel's change bounded by $\epsilon$.

**Results to expect:**

- Accuracy on adversarial images drops sharply.
- **Adversarial training** (training on clean plus FGSM images) improves robustness to perturbations and acts as a regularizer. It may cost a little clean accuracy.

---

### B5. Saving, loading, and fine-tuning

- **`state_dict`:** an OrderedDict mapping each layer name to its tensor, for example `conv1.weight`, `conv1.bias`, …, `fc3.bias`.
  - BatchNorm also stores `running_mean`, `running_var`, and `num_batches_tracked`.
  - Optimizers have their own `state_dict` too, holding momentum buffers.

**Save and load:**

```python
torch.save(net.state_dict(), 'pokemon_net.pth')
net2 = Net(); net2.load_state_dict(torch.load('pokemon_net.pth')); net2.eval()
```

**Partial loading (fine-tuning on CIFAR-10):** load everything except fc3, so the classifier head starts fresh.

```python
pretrained = torch.load('pokemon_net.pth')
subset = {k: v for k, v in pretrained.items() if k in selected_layers}
net_cifar.load_state_dict(subset, strict=False)   # fc3 stays randomly initialized
```

- **Why it works:** early conv layers learn generic features (edges, colors, textures) that transfer across datasets. Only the head is task-specific.
- **Optional:** freeze the transferred layers with `p.requires_grad = False`, or give them a lower learning rate.
- **Note:** CIFAR-10 images are 32×32 but get resized to 64×64. That keeps the flatten size at $32 \cdot 16 \cdot 16$, so fc1's weights still fit.

---

### B6. Neural style transfer (Gatys et al.)

**Core idea:** the network weights are frozen; you optimize the **pixels of the output image**.

- **Feature extractor:** pretrained VGG-19 (`models.vgg19(pretrained=True).features.to(device).eval()`).
- **Starting point:** a copy of the content image.

#### Content loss

Let $F^l$ be the feature map at layer $l$ for the generated image and $P^l$ for the content image:

$$
\mathcal{L}_{\text{content}} = \text{MSE}\left(F^l, P^l\right)
$$

#### Style loss via the Gram matrix

Reshape the feature map of shape $a \times b \times c \times d$ (batch × channels × H × W) into $F \in \mathbb{R}^{ab \times cd}$. Then:

$$
G = \frac{F F^{T}}{a \cdot b \cdot c \cdot d}, \qquad \mathcal{L}_{\text{style}} = \sum_{l} \text{MSE}\left(G^l_{\text{gen}}, G^l_{\text{style}}\right)
$$

- **What the Gram matrix measures:** $G_{ij}$ is the correlation between channels $i$ and $j$ summed over all positions. Spatial layout is discarded and only texture remains.
- *Analogy:* it is like a classical guitar recording's timbre. It captures which overtones occur together, not when each note is played.

**Loss modules** pass the input through unchanged and store the loss as a side effect:

```python
class ContentLoss(nn.Module):
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target); return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__(); self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        self.loss = F.mse_loss(gram_matrix(input), self.target); return input
```

**VGG normalization** (ImageNet statistics, applied before the image enters VGG):

```python
return (img - self.mean.view(-1,1,1)) / self.std.view(-1,1,1)
```

**Total loss:**

$$
\mathcal{L} = \alpha \mathcal{L}_{\text{content}} + \beta \mathcal{L}_{\text{style}}
$$

In the homework, $\alpha = 1$ and $\beta = 10^6$ by default.

**Optimization loop body:**

```python
optimizer.zero_grad()
model(input_img)
style_score = style_weight * sum(sl.loss for sl in style_losses)
content_score = content_weight * sum(cl.loss for cl in content_losses)
(style_score + content_score).backward()
optimizer.step()
```

#### Expected observations

| Experiment | Result |
|---|---|
| Style weight $10^6$ | Strong painterly texture; content partly distorted |
| Style weight 5000 | Balanced: content clear, mild style |
| Style weight 10 | Looks almost like the original content photo |
| Style from conv_1 | Fine, low-level texture (colors, small strokes) |
| Style from conv_3 | Mid-level patterns (brush strokes, motifs) |
| Style from conv_5 | Larger, more abstract structures (swirls) |

The usual practice is to combine several style layers to capture multi-scale texture. Content comes from a mid-depth layer such as conv_4, which preserves structure without exact pixels.

**Gotchas:**

- `.detach()` the targets so they are treated as constants.
- Clamp the image to $[0, 1]$ during optimization.
- Truncate VGG after the last loss layer to save compute.
- The paper uses L-BFGS; Adam also works.

---

## Part C: Rapid-fire self-test

1. Give 2-ReLU weights that approximate a step reaching 1 at $\delta$.
2. Why can't $\sigma(x) = x^2 + x + 1$ with one hidden layer approximate $\sin x$?
3. Write $\sigma'(x)$ in terms of $\sigma$. What is its maximum?
4. For $Z = XW$, what are $\partial L / \partial W$ and $\partial L / \partial X$?
5. Gradient of $\frac{1}{2N}\sum (y - \hat{y})^2$ with respect to $\hat{y}$?
6. Why is `q.grad` `None` in the Q4.1 autograd example?
7. Output size of a conv with $W = 64$, $K = 5$, $P = 0$, $S = 2$?
8. Parameter count of `Conv2d(16, 32, 3)`?
9. Name one difference between BatchNorm and LayerNorm.
10. What does the Gram matrix capture, and why does it discard spatial layout?
11. In style transfer, what is being optimized?
12. Write the FGSM update.
13. How do you load only some layers from a `state_dict`?

**Answers:**

1. $a = \left[\frac{1}{\delta}, -\frac{1}{\delta}\right]$, $b = [1, 1]$, $c = [0, -\delta]$.
2. The output is always a quadratic polynomial, regardless of width.
3. $\sigma(1-\sigma)$; maximum $0.25$.
4. $X^{T} G$ and $G W^{T}$.
5. $(\hat{y} - y)/N$.
6. It is a non-leaf tensor, and gradients of non-leaf tensors are not retained.
7. $\lfloor 59/2 \rfloor + 1 = 30$.
8. $16 \cdot 32 \cdot 9 + 32 = 4640$.
9. BatchNorm uses batch statistics per channel; LayerNorm uses per-sample statistics. BatchNorm behaves differently in train and eval.
10. Channel co-activations (texture); the dot product sums over all positions.
11. The image pixels; VGG is frozen.
12. $x + \epsilon \cdot \text{sign}(\nabla_x J)$.
13. Filter the dict keys, then call `load_state_dict(..., strict=False)`.
