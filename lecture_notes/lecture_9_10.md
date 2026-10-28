# Convolutional Neural Networks (CNN): Lecture Notes

> EC523 Fall 2026, CNNs + architecture (Oct 14–21)
> Credits: Prof. Kosta Derpanis, *Introduction to Computer Vision* (EECS 4422); Brian Kulis, *Deep Learning 2023*; Fei-Fei Li, Andrej Karpathy & Justin Johnson (AlexNet slides)

---

## Table of Contents

1. [Basic Math: Image Noise](#1-basic-math-image-noise)
2. [Denoising by Averaging](#2-denoising-by-averaging)
3. [Convolution](#3-convolution)
4. [From Fully Connected to Convolutional Layers](#4-from-fully-connected-to-convolutional-layers)
5. [Building Blocks of a CNN](#5-building-blocks-of-a-cnn)
6. [Receptive Field and Dilated Convolution](#6-receptive-field-and-dilated-convolution)
7. [Pooling Layers](#7-pooling-layers)
8. [Activation Functions](#8-activation-functions)
9. [Putting It Together](#9-putting-it-together)
10. [Case Study: AlexNet](#10-case-study-alexnet)
11. [The Revolution of Depth](#11-the-revolution-of-depth)
12. [Quick Reference](#12-quick-reference)

---

## 1. Basic Math: Image Noise

A captured image is modeled as an ideal (clean) image plus additive noise:

```math
I(x, y) = I_{\text{ideal}}(x, y) + \eta(x, y)
```

where the noise is Gaussian with zero mean:

```math
\eta(x, y) \sim \mathcal{N}(\mu = 0, \sigma^2)
```

**Key assumption:** the noise at each pixel is **independent and identically distributed (i.i.d.)**.

The Gaussian probability density function:

```math
p(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{x^2}{2\sigma^2} \right)
```

### Wisdom of the crowd

Because the noise is zero-mean and independent, averaging many noisy observations cancels the noise out. Averaging $N$ i.i.d. samples reduces the noise variance by a factor of $N$:

$$
\mathrm{Var}\left( \frac{1}{N} \sum_{i=1}^{N} \eta_i \right) = \frac{\sigma^2}{N}
$$

> **Example (classical guitar recording):** Room hiss on a mic is roughly i.i.d. noise. If you record the same sustained harmonic many times and average the aligned takes, the note stays while the hiss shrinks, by a factor of $\sqrt{N}$ in amplitude.

---

## 2. Denoising by Averaging

**How can we remove the noise?**

Assumptions:
- Noise is **i.i.d.**
- **Neighboring pixels are similar** (the true signal varies slowly)

**Idea:** replace each pixel with the **average of its neighbors** (a moving average). The result is a *smoothed* signal: jagged spikes are reduced while the overall shape is kept.

### 2D average example (3 × 3 window)

Input:

```math
\begin{bmatrix}
0 & 0 & 0 & 9 & 18 \\
0 & 90 & 0 & 27 & 0 \\
0 & 0 & 0 & 9 & 0 \\
0 & 0 & 0 & 9 & 0 \\
0 & 0 & 0 & 0 & 18
\end{bmatrix}
```

Each output pixel = (sum of the 3 × 3 neighborhood) / 9. For example, at row 1, column 1:

```math
\frac{0+0+0+0+90+0+0+0+0}{9} = 10
```

At row 1, column 2:

```math
\frac{0+0+9+90+0+27+0+0+9}{9} = \frac{135}{9} = 15
```

Full output for the interior (valid) positions:

```math
\begin{bmatrix}
10 & 15 & 7 \\
10 & 15 & 5 \\
0 & 2 & 4
\end{bmatrix}
```

(Values rounded to integers.) Note how the outlier $90$ gets spread across its neighbors. This moving average *is* a convolution with the kernel:

```math
G = \frac{1}{9}
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1
\end{bmatrix}
```

---

## 3. Convolution

### Definition

For an image $F$ and a kernel (filter) $G$ of size $(2K+1) \times (2K+1)$:

```math
H[x, y] = \sum_{u=-K}^{K} \sum_{v=-K}^{K} G[u, v] \; F[x - u, y - v]
```

Written compactly:

```math
H = G \ast F
```

### Impulse response

Convolving an image that is a single $1$ in a field of zeros (an *impulse*) with kernel

```math
G = \begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix}
```

reproduces the kernel itself. Because true convolution **flips** the kernel (indices $x-u$ and $y-v$), the kernel appears upside-down and mirrored when overlaid on the image, and the output reads $a, b, c, \ldots$ in normal order.

### Properties

**Commutative**

```math
G \ast F = F \ast G
```

**Distributive**

```math
E \ast (F + G) = (E \ast F) + (E \ast G)
```

**Linearity**: if $H_1 = G \ast F_1$ and $H_2 = G \ast F_2$, then

```math
G \ast (\alpha F_1 + \beta F_2) = \alpha H_1 + \beta H_2
```

**Shift invariance**: shifting the input shifts the output by the same amount

```math
G[x, y] \ast F[x - \alpha, y - \beta] = H[x - \alpha, y - \beta]
```

> **Example (soccer):** A filter that detects "ball" responds the same way whether the ball is in the penalty box or at the center circle. Shift invariance means one detector works everywhere on the pitch.

### Convolution vs. cross-correlation

Deep learning libraries (e.g. PyTorch `nn.Conv2d`) actually compute **cross-correlation**, with no kernel flip:

```math
H[x, y] = \sum_{u} \sum_{v} G[u, v] \; F[x + u, y + v]
```

Since the kernel weights are *learned*, the flip makes no practical difference, and the field still calls it "convolution". The worked examples below use this (no-flip) version.

### Example: vertical edge filter

```math
\underbrace{\begin{bmatrix}
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10 \\
0 & 0 & 0 & 10 & 10 & 10
\end{bmatrix}}_{6 \times 6}
\ast
\underbrace{\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}}_{3 \times 3}
=
\underbrace{\begin{bmatrix}
0 & -30 & -30 & 0 \\
0 & -30 & -30 & 0 \\
0 & -30 & -30 & 0 \\
0 & -30 & -30 & 0
\end{bmatrix}}_{4 \times 4}
```

The output is non-zero only where the intensity changes from left to right, i.e. at the vertical edge. For one window straddling the edge:

```math
3 \times (1 \cdot 0 + 0 \cdot 0 + (-1) \cdot 10) = -30
```

### Convolution with multiple input channels

For an input with $C$ channels (e.g. RGB, $C = 3$), the kernel also has $C$ slices. Each slice is convolved with its own channel, and the **results are summed** into a single output map:

```math
H[x, y] = \sum_{c=1}^{C} \left( G_c \ast F_c \right)[x, y]
```

Slide example (top-left output): channel results $7$, $5$, $7$ give $7 + 5 + 7 = 19$.

### Convolution as template matching ("Where's Waldo?")

Using an image patch of Waldo as the filter, the output map has its **strongest response** where the image best matches the filter, so the peak marks where Waldo is.

---

## 4. From Fully Connected to Convolutional Layers

### Problem with fully connected (FC) layers

A $768 \times 1024$ image flattened into a vector:

```math
\mathbf{x} \in \mathbb{R}^{786432 \times 1}, \qquad \mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b} \in \mathbb{R}^{K \times 1}
```

Every hidden unit connects to every pixel, giving **~800,000 parameters per hidden unit**. That is far too many, and it ignores the spatial structure of images.

### Inspiration from the visual cortex

- Neurons are **spatially localized**
- **Topographic** feature maps
- **Hierarchical** feature processing

### Two key ideas of a conv layer

1. **Local connectivity:** each output unit is a linear function of a **localized subset** of input units (e.g. a $3 \times 3$ patch).
2. **Weight sharing:** the **same linear transformation is applied at each position** (shift invariance). So the number of parameters does not depend on image size.

---

## 5. Building Blocks of a CNN

### Convolution as a pointwise multiply-and-sum

Input $I$ and kernel $F$:

```math
I = \begin{bmatrix}
3 & 3 & 2 & 1 & 0 \\
0 & 0 & 1 & 3 & 1 \\
3 & 1 & 2 & 2 & 3 \\
2 & 0 & 0 & 2 & 2 \\
2 & 0 & 0 & 0 & 1
\end{bmatrix},
\qquad
F = \begin{bmatrix}
0 & 1 & 2 \\
2 & 2 & 0 \\
0 & 1 & 2
\end{bmatrix}
```

Top-left output (elementwise product, then sum):

```math
3 \cdot 0 + 3 \cdot 1 + 2 \cdot 2 + 0 \cdot 2 + 0 \cdot 2 + 1 \cdot 0 + 3 \cdot 0 + 1 \cdot 1 + 2 \cdot 2 = 12
```

**Stride 1** output ($3 \times 3$):

```math
\begin{bmatrix}
12 & 12 & 17 \\
10 & 17 & 19 \\
9 & 6 & 14
\end{bmatrix}
```

**Stride 2** (the window jumps 2 pixels at a time) output ($2 \times 2$):

```math
\begin{bmatrix}
12 & 17 \\
9 & 14
\end{bmatrix}
```

### Output size formula

For input width $N$, kernel size $K$, padding $P$, stride $S$:

```math
N_{\text{out}} = \left\lfloor \frac{N - K + 2P}{S} \right\rfloor + 1
```

Checks:
- Stride 1: $(5 - 3 + 0)/1 + 1 = 3$
- Stride 2: $(5 - 3 + 0)/2 + 1 = 2$
- AlexNet CONV1: $(227 - 11 + 0)/4 + 1 = 55$

### Multiple filters → multiple output channels

- One filter of size $K \times K \times C_{\text{in}}$ produces **one** output channel.
- Using $C_{\text{out}}$ filters produces an output with $C_{\text{out}}$ channels (stacked feature maps).

### How many parameters in this layer?

Input $32 \times 32 \times 3$, three filters of size $3 \times 3 \times 3$, output $32 \times 32 \times 3$:

```math
\underbrace{3 \times 3 \times 3}_{\text{per filter}} \times \underbrace{3}_{\text{filters}} = 81 \text{ weights}, \qquad 81 + 3 \text{ biases} = 84 \text{ parameters}
```

General formula:

```math
\#\text{params} = (K \times K \times C_{\text{in}} + 1) \times C_{\text{out}}
```

Compare with a fully connected layer mapping $32 \times 32 \times 3 \to 32 \times 32 \times 3$: $3072 \times 3072 \approx 9.4$ million weights.

---

## 6. Receptive Field and Dilated Convolution

### Receptive field

The **receptive field** of a unit is the region of the *original input image* that affects it. Stacking conv (and pooling) layers makes deeper units "see" larger regions: Layer 1 sees a tiny patch, Layer 2 a larger one, Layer 3 larger still.

For $L$ stacked $K \times K$ conv layers with stride 1:

```math
\text{RF} = 1 + L \times (K - 1)
```

e.g. three $3 \times 3$ layers give a $7 \times 7$ receptive field.

### Dilated (atrous) convolution

Insert gaps between kernel taps: a $3 \times 3$ kernel with **dilation factor** $d$ samples pixels $d$ apart.

```math
K_{\text{eff}} = K + (K - 1) \times (d - 1)
```

| Dilation $d$ | Effective size of a $3 \times 3$ kernel |
|---|---|
| 1 (standard) | $3 \times 3$ |
| 2 | $5 \times 5$ |
| 3 | $7 \times 7$ |

Benefits:
- **Increases spatial filter context** (larger receptive field)
- **Parameter efficient** (still only 9 weights)
- **Computationally efficient**

---

## 7. Pooling Layers

Pooling summarizes a local neighborhood, reducing spatial size and adding some robustness to small shifts. It has **no learnable parameters**.

Example input ($4 \times 4$), 2 × 2 pooling with stride 2:

```math
\begin{bmatrix}
1 & 1 & 4 & 2 \\
7 & 3 & 9 & 5 \\
1 & 1 & 7 & 3 \\
4 & 2 & 8 & 6
\end{bmatrix}
```

### Average pooling

```math
h_k(x, y, c) = \frac{1}{\lvert \mathcal{N}(x, y) \rvert} \sum_{(i, j) \in \mathcal{N}(x, y)} h_{k-1}(i, j, c)
```

Result:

```math
\begin{bmatrix} 3 & 5 \\ 2 & 6 \end{bmatrix}
```

e.g. top-left: $(1 + 1 + 7 + 3)/4 = 3$

### Max pooling

```math
h_k(x, y, c) = \max_{(i, j) \in \mathcal{N}(x, y)} h_{k-1}(i, j, c)
```

Result:

```math
\begin{bmatrix} 7 & 9 \\ 4 & 8 \end{bmatrix}
```

> **Example (F1 telemetry):** Max pooling over each track sector is like reporting each sector's top speed: you keep the strongest reading and don't care exactly where in the sector it happened.

Pooling is applied **per channel**, so the number of channels is unchanged.

### Channel-wise pooling: 1 × 1 convolution

A $1 \times 1$ convolution maps $H \times W \times D \to H \times W \times D'$:

- **Does not change the spatial dimension**, but **can change the channel dimension**.
- Each output pixel is a weighted combination of the $D$ channel values at that same location, so it can be viewed as **channel pooling**.
- Typically used to **reduce over-fitting** and **compress the network** (fewer channels → fewer parameters in later layers).

```math
\#\text{params} = (1 \times 1 \times D + 1) \times D'
```

---

## 8. Activation Functions

Non-linearities applied element-wise after the linear (conv) step. Without them, a stack of conv layers collapses into one linear operation.

**Hyperbolic tangent**: range $(-1, 1)$, zero-centered

```math
y(x) = \tanh(x)
```

**Sigmoid**: range $(0, 1)$, $\sigma(0) = 0.5$

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

**Rectified Linear Unit (ReLU)**: cheap, does not saturate for $x > 0$

```math
y(x) = \max(0, x)
```

**Exponential Linear Unit (ELU)**: smooth negative part saturating at $-\alpha$

```math
y(x) =
\begin{cases}
x & x > 0 \\
\alpha (e^{x} - 1) & x \le 0
\end{cases}
```

---

## 9. Putting It Together

A CNN repeats a basic block, then ends with fully connected layers that output class scores (e.g. "cat"):

```
input ──► [ linear filtering ──► non-linear activation ──► spatial pooling ] × N ──► FC layers ──► class scores
              (convolution)          (e.g. ReLU)              (max / avg)
```

Each intermediate feature map is a 3D tensor with dimensions $x$ (width), $y$ (height), and **channels**. As we go deeper, spatial size usually **shrinks** while the number of channels **grows**.

---

## 10. Case Study: AlexNet

*"ImageNet Classification with Deep Convolutional Neural Networks", Krizhevsky et al., 2012*

- First time a deep CNN **won the ILSVRC** (ImageNet) challenge.
- **Top-5 test error 15.4%** vs. 26.2% for the next best entry.
- Shocked the computer vision community and started the deep learning "wave".

> **Top-5 error:** the fraction of test images where the correct label is *not* among the model's top 5 predictions.

### Architecture (simplified)

| Output size | Layer | Details |
|---|---|---|
| 227 × 227 × 3 | INPUT | |
| 55 × 55 × 96 | CONV1 | 96 filters 11 × 11, stride 4, pad 0 |
| 27 × 27 × 96 | MAX POOL1 | 3 × 3, stride 2 |
| 27 × 27 × 96 | NORM1 | normalization layer |
| 27 × 27 × 256 | CONV2 | 256 filters 5 × 5, stride 1, pad 2 |
| 13 × 13 × 256 | MAX POOL2 | 3 × 3, stride 2 |
| 13 × 13 × 256 | NORM2 | normalization layer |
| 13 × 13 × 384 | CONV3 | 384 filters 3 × 3, stride 1, pad 1 |
| 13 × 13 × 384 | CONV4 | 384 filters 3 × 3, stride 1, pad 1 |
| 13 × 13 × 256 | CONV5 | 256 filters 3 × 3, stride 1, pad 1 |
| 6 × 6 × 256 | MAX POOL3 | 3 × 3, stride 2 |
| 4096 | FC6 | 4096 neurons |
| 4096 | FC7 | 4096 neurons |
| 1000 | FC8 | 1000 neurons (class scores) |

Sanity checks with the output-size formula:

```math
\text{CONV1: } \frac{227 - 11}{4} + 1 = 55, \qquad \text{POOL1: } \frac{55 - 3}{2} + 1 = 27
```

CONV1 parameter count:

```math
(11 \times 11 \times 3 + 1) \times 96 = 34{,}944
```

### Details / retrospectives

- First use of **ReLU**
- Used **Norm layers** (not common anymore)
- **Heavy data augmentation**
- **Dropout** 0.5
- Batch size 128
- **SGD with momentum** 0.9
- Learning rate $10^{-2}$, reduced by $10\times$ manually when validation accuracy plateaus
- **L2 weight decay** $5 \times 10^{-4}$
- **Ensemble of 7 CNNs:** 18.2% → 15.4% error

---

## 11. The Revolution of Depth

ImageNet classification top-5 error (%):

| Year | Model | Depth | Top-5 error |
|---|---|---|---|
| ILSVRC'10 | — | shallow | 28.2 |
| ILSVRC'11 | — | shallow | 25.8 |
| ILSVRC'12 | **AlexNet** | 8 layers | 16.4 |
| ILSVRC'13 | — | 8 layers | 11.7 |
| ILSVRC'14 | **VGG** | 19 layers | 7.3 |
| ILSVRC'14 | **GoogLeNet** | 22 layers | 6.7 |
| ILSVRC'15 | **ResNet** | **152 layers** | **3.57** |

*Source: He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition", CVPR 2016.*

### Architectures at a glance

- **AlexNet (8 layers):** 11 × 11 conv 96 /4 + pool → 5 × 5 conv 256 + pool → 3 × 3 conv 384 → 3 × 3 conv 384 → 3 × 3 conv 256 + pool → FC 4096 → FC 4096 → FC 1000
- **VGG (19 layers):** only **3 × 3 convs**, stacked in groups (64, 64 → 128, 128 → 256 × 4 → 512 × 4 → 512 × 4), each group ending with pool/2, then FC 4096 → FC 4096 → FC 1000
- **GoogLeNet (22 layers):** "Inception" modules with parallel branches of different filter sizes, concatenated
- **ResNet (152 layers):** very deep stack of blocks with **skip (residual) connections**

### Spectrum of depth: what it takes to train deeper nets

| Depth | What's needed |
|---|---|
| ~5 layers | easy |
| > 10 layers | good **initialization**, **Batch Normalization** |
| > 30 layers | **skip connections** |
| > 100 layers | **identity skip connections** |
| > 1000 layers | ? (open question) |

---

## 12. Quick Reference

| Concept | Formula |
|---|---|
| Noise model | $I = I_{\text{ideal}} + \eta$ with $\eta \sim \mathcal{N}(0, \sigma^2)$ i.i.d. |
| 2D convolution | $H[x, y] = \sum_{u} \sum_{v} G[u, v] F[x - u, y - v]$ |
| Output size | $\lfloor (N - K + 2P)/S \rfloor + 1$ |
| Conv layer params | $(K \times K \times C_{\text{in}} + 1) \times C_{\text{out}}$ |
| Receptive field (stride 1) | $1 + L \times (K - 1)$ |
| Dilated kernel size | $K + (K - 1) \times (d - 1)$ |
| Max pooling | $\max$ over each neighborhood, per channel |
| Average pooling | mean over each neighborhood, per channel |
| ReLU | $\max(0, x)$ |
| Sigmoid | $1 / (1 + e^{-x})$ |

### Key takeaways

1. Averaging neighbors removes i.i.d. noise; that averaging is a convolution.
2. Convolution is **linear** and **shift-invariant**; CNNs learn the kernels.
3. Conv layers use **local connectivity** + **weight sharing**, so they need vastly fewer parameters than FC layers.
4. A CNN block = **conv → non-linearity → pooling**, repeated.
5. Depth wins (AlexNet → VGG → GoogLeNet → ResNet), but very deep nets need BatchNorm, good init, and skip connections.
