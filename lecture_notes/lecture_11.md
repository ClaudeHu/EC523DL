# Lecture 11 — CNN II & Advanced Architectures

> EC523 Deep Learning · Oct 19
> Topics: Neural Style Transfer · Modern CNNs (AlexNet, VGG, GoogLeNet) · ResNet (residual learning, identity mappings, pre-activation)

---

## Table of Contents

1. [Logistics](#1-logistics)
2. [Recap: CNNs](#2-recap-cnns)
3. [Neural Style Transfer](#3-neural-style-transfer)
4. [Modern CNNs](#4-modern-cnns)
   - [ImageNet / ILSVRC](#41-imagenet--ilsvrc)
   - [AlexNet](#42-alexnet-2012)
   - [VGGNet](#43-vggnet-2014)
   - [GoogLeNet](#44-googlenet-2014)
5. [ResNet](#5-resnet-deep-residual-networks)
   - [Spectrum of depth](#51-spectrum-of-depth)
   - [Initialization](#52-initialization-revisited)
   - [Batch normalization](#53-batch-normalization)
   - [Degradation problem](#54-the-degradation-problem)
   - [Residual learning](#55-residual-learning)
   - [Smooth forward / backward propagation](#56-why-it-works-smooth-forward--backward-propagation)
   - [Bottleneck design](#57-bottleneck-block)
   - [Representation vs optimization vs generalization](#58-representation-optimization-generalization)
6. [Identity Mappings: 100 → 1000 layers](#6-importance-of-the-identity-from-100-to-1000-layers)
7. [Summary](#7-summary)

---

## 1. Logistics

- HW3 released.
- Short Quiz 2 — **Oct 28** (15 min, covers HW2–HW3).
- Midway proposal due — **Nov 2**.
- Short Quiz 3 — Dec 2 (covers HW4–HW5).
- Presentation day — **Dec 10** (evening, with pizza).

| Dates | Topic |
|---|---|
| Sep 2–16 | Course overview + ML review |
| Sep 21–30 | Neural network basics |
| Oct 5–13 | Training |
| Oct 14–21 | CNNs + architectures |
| Oct 26–Nov 11 | Unsupervised + generative models |
| Nov 16–23 | RNNs + transformers |
| Nov 30–Dec 7 | Self-supervision · GNNs · explainability |

---

## 2. Recap: CNNs

- Convolution is good at finding **local features** (weight sharing + locality).
- Useful for vision, audio, and even text.
- LeNet was the first classic CNN.

---

## 3. Neural Style Transfer

> Gatys, Ecker & Bethge, *"Image Style Transfer Using Convolutional Neural Networks"*, CVPR 2016.

**Goal:** `content image + style image = stylized image`
(e.g. a selfie rendered in the brush-stroke style of a Van Gogh self-portrait).

### 3.1 Two stages

| Stage | What happens |
|---|---|
| **Analysis** | Pass the content image and the style image through a **pretrained CNN** (trained for object recognition, e.g. VGG) and record features at chosen layers. |
| **Synthesis** | Start from an image (noise or the content image) and **iteratively adjust the input pixels** by gradient descent so its features match the content and style features. The network weights stay **frozen**. |

### 3.2 Content representation

Content = **raw convolutional activations** of layer $`\ell`$ from the content image:

```math
\mathbf{A}^{[\ell]}(I) \in \mathbb{R}^{H \times W \times C}
```

Deeper layers capture *what* is in the image and *where*, but not exact pixel values.

### 3.3 Style representation — the Gram matrix

Take the activation volume of layer $`\ell`$ (size $`H \times W \times C`$). At each spatial location $`(i, j)`$ there is a channel vector $`\mathbf{v}_{i,j} \in \mathbb{R}^{C}`$.

The **Gram matrix** is $`C \times C`$ (channels × channels):

```math
G_{k,k'} = \sum_{i=1}^{\text{rows}} \sum_{j=1}^{\text{cols}} v_{i,j,k} \, v_{i,j,k'}
```

Equivalently, reshape the activations into $`\mathbf{V} \in \mathbb{R}^{(HW) \times C}`$, then

```math
\mathbf{G} = \mathbf{V}^{\top} \mathbf{V}
```

- Entry $`G_{k,k'}`$ measures how strongly channels $`k`$ and $`k'`$ **fire together**.
- Summing over all positions **throws away spatial layout** → keeps texture/color/brush-stroke statistics = *style*.

> 🎸 **Intuition:** think of each channel as one guitar string. The Gram matrix records *which strings tend to ring together* (the harmony / timbre of a piece), but not *when* each note was played (the spatial arrangement). Two performances with the same "sound" but different melodies have similar Gram matrices.

### 3.4 Style transfer loss

```math
\mathcal{L}(I_{\text{generated}}) =
\lambda_1 \left\lVert \mathbf{A}^{[\ell]}(I_{\text{content}}) - \mathbf{A}^{[\ell]}(I_{\text{generated}}) \right\rVert_F^2
+
\lambda_2 \left\lVert \mathbf{G}^{[\ell]}(I_{\text{style}}) - \mathbf{G}^{[\ell]}(I_{\text{generated}}) \right\rVert_F^2
```

- First term: **content loss** (match activations).
- Second term: **style loss** (match Gram matrices; in practice summed over several layers).
- $`\lambda_1 / \lambda_2`$ trades off content fidelity vs. stylization strength.
- Optimize w.r.t. **the pixels of** $`I_{\text{generated}}`$.

### 3.5 Variants

- **Artistic style transfer** — painting style onto a photo.
- **Photorealistic style transfer** — e.g. transfer a night-time city look onto a daytime city photo while keeping it photo-like.

---

## 4. Modern CNNs

### Brief history

- **1990s** — AT&T builds a convnet for reading checks.
- **2000s** — OCR & handwriting recognition deployed (e.g. Microsoft).
- **2012** — Convnets win ImageNet.
- **2013+** — Spike in commercial interest in deep learning.
- Convnets were among the first deep nets successfully trained with backprop, probably thanks to their efficiency (weight sharing).

### 4.1 ImageNet / ILSVRC

"The Olympics of computer vision" — a subset of ImageNet:

- 1,000 object classes
- 1.2M training images
- 100K test images

**Top-5 error:** the model fails if the correct label is not among its 5 highest-scoring predictions.

#### Revolution of depth (ILSVRC top-5 error %)

| Year | Model | Depth | Top-5 error (%) |
|---|---|---|---|
| 2010 | (shallow) | — | 28.2 |
| 2011 | (shallow) | — | 25.8 |
| 2012 | **AlexNet** | 8 | 16.4 |
| 2013 | ZFNet | 8 | 11.7 |
| 2014 | **VGG** | 19 | 7.3 |
| 2014 | **GoogLeNet** | 22 | 6.7 |
| 2015 | **ResNet** | **152** | **3.57** |

Similar trend for PASCAL VOC 2007 object detection (mAP): HOG/DPM 34 → AlexNet (R-CNN) 58 → VGG (R-CNN) 66 → ResNet-101 (Faster R-CNN) 86.

### 4.2 AlexNet (2012)

> Krizhevsky, Sutskever & Hinton, *"ImageNet Classification with Deep Convolutional Neural Networks"*, 2012.

- First deep CNN to win ILSVRC.
- Top-5 error **15.4%** vs. **26.2%** for the runner-up → started the deep learning "wave".

#### Architecture (simplified)

| Layer | Output size | Details |
|---|---|---|
| INPUT | 227×227×3 | |
| CONV1 | 55×55×96 | 96 filters 11×11, stride 4, pad 0 |
| MAX POOL1 | 27×27×96 | 3×3, stride 2 |
| NORM1 | 27×27×96 | local response normalization |
| CONV2 | 27×27×256 | 256 filters 5×5, stride 1, pad 2 |
| MAX POOL2 | 13×13×256 | 3×3, stride 2 |
| NORM2 | 13×13×256 | |
| CONV3 | 13×13×384 | 384 filters 3×3, stride 1, pad 1 |
| CONV4 | 13×13×384 | 384 filters 3×3, stride 1, pad 1 |
| CONV5 | 13×13×256 | 256 filters 3×3, stride 1, pad 1 |
| MAX POOL3 | 6×6×256 | 3×3, stride 2 |
| FC6 | 4096 | |
| FC7 | 4096 | |
| FC8 | 1000 | class scores |

Output spatial size of a conv/pool layer (input width $`W`$, kernel $`K`$, padding $`P`$, stride $`S`$):

```math
W_{\text{out}} = \frac{W - K + 2P}{S} + 1
\qquad \text{e.g. CONV1: } \frac{227 - 11 + 0}{4} + 1 = 55
```

#### Details / retrospective

- **First use of ReLU**
- Norm layers (no longer common)
- Heavy data augmentation
- Dropout 0.5
- Batch size 128
- SGD with momentum 0.9
- Learning rate $`10^{-2}`$, divided by 10 manually when val accuracy plateaus
- L2 weight decay $`5 \times 10^{-4}`$
- 7-CNN ensemble: 18.2% → 15.4%

### 4.3 VGGNet (2014)

> Simonyan & Zisserman, 2014.

- **Only** 3×3 CONV (stride 1, pad 1) and 2×2 MAX POOL (stride 2) — a very simple, uniform design.
- Top-5 error improved from 11.2% (best ILSVRC 2013 model) to **7.3%**.
- Why stacks of 3×3? Two 3×3 convs have the same receptive field as one 5×5 (three ≈ one 7×7), with **fewer parameters** and **more non-linearities**.

#### VGG-16 memory & parameter budget

Params of a conv layer $`= (K \times K \times C_{\text{in}}) \times C_{\text{out}}`$ (biases ignored).

| Layer | Output | Memory (activations) | Params |
|---|---|---|---|
| INPUT | 224×224×3 | 150K | 0 |
| CONV3-64 | 224×224×64 | 3.2M | (3·3·3)·64 = 1,728 |
| CONV3-64 | 224×224×64 | 3.2M | (3·3·64)·64 = 36,864 |
| POOL2 | 112×112×64 | 800K | 0 |
| CONV3-128 | 112×112×128 | 1.6M | (3·3·64)·128 = 73,728 |
| CONV3-128 | 112×112×128 | 1.6M | (3·3·128)·128 = 147,456 |
| POOL2 | 56×56×128 | 400K | 0 |
| CONV3-256 ×3 | 56×56×256 | 800K each | 294,912 / 589,824 / 589,824 |
| POOL2 | 28×28×256 | 200K | 0 |
| CONV3-512 ×3 | 28×28×512 | 400K each | 1,179,648 / 2,359,296 / 2,359,296 |
| POOL2 | 14×14×512 | 100K | 0 |
| CONV3-512 ×3 | 14×14×512 | 100K each | 2,359,296 each |
| POOL2 | 7×7×512 | 25K | 0 |
| FC | 4096 | 4096 | 7·7·512·4096 = **102,760,448** |
| FC | 4096 | 4096 | 4096·4096 = 16,777,216 |
| FC | 1000 | 1000 | 4096·1000 = 4,096,000 |

- **Total memory:** ≈ 24M activations × 4 bytes ≈ **93 MB / image** (forward only; ≈ ×2 for backward).
- **Total params:** ≈ **138M**.

> 📌 **Key observation:** Most **memory** is in the **early CONV** layers; most **parameters** are in the **late FC** layers.

### 4.4 GoogLeNet (2014)

> Szegedy et al., 2014 — **ILSVRC 2014 winner (6.7% top-5)**.

#### Inception module

Parallel branches applied to the previous layer, then **concatenated along channels**:

```
                 Previous layer
     ┌──────────┬──────┴───────┬──────────────┐
  1x1 conv   1x1 conv      1x1 conv     3x3 max pool
     │          │              │              │
     │      3x3 conv       5x5 conv       1x1 conv
     └──────────┴──────┬───────┴──────────────┘
               Filter concatenation
```

- The 1×1 convs act as **bottlenecks** that reduce channel depth before the expensive 3×3 / 5×5 convs.
- Multi-scale features in a single block.

#### Fun facts

- Only **5M parameters** — removes the big FC layers entirely (global average pooling).
- Compared with AlexNet: **12× fewer params**, ~2× more compute, 6.67% vs 16.4% error.

---

## 5. ResNet: Deep Residual Networks

> He, Zhang, Ren & Sun, *"Deep Residual Learning for Image Recognition"*, CVPR 2016.
> Slides adapted from Kaiming He's tutorial.

- **ILSVRC 2015 winner — 3.57% top-5 error** with "ultra-deep" 152-layer nets.
- 1st place in all five main ILSVRC & COCO 2015 tracks:
  - ImageNet Detection: 16% better than 2nd
  - ImageNet Localization: 27% better than 2nd
  - COCO Detection: 11% better than 2nd
  - COCO Segmentation: 12% better than 2nd
  *(relative improvements)*
- Simple, clean, easy to reproduce; a backbone for classification, detection, segmentation, …

### 5.1 Spectrum of depth

| Depth | What you need |
|---|---|
| ~5 layers | easy |
| > 10 layers | good **initialization**, **Batch Normalization** |
| > 30 layers | **skip connections** |
| > 100 layers | **identity** skip connections |
| > 1000 layers | ? (→ pre-activation ResNet, Section 6) |

### 5.2 Initialization (revisited)

For a linear layer $`y = Wx`$ with $`n^{\text{in}}`$ inputs, assuming linear activation and $`x, w, y`$ independent:

```math
\operatorname{Var}(y) = \left( n^{\text{in}} \operatorname{Var}(w) \right) \operatorname{Var}(x)
```

Over multiple layers $`d = 1, \dots, D`$:

```math
\text{Forward: } \operatorname{Var}(y) = \left( \prod_{d} n_d^{\text{in}} \operatorname{Var}(w_d) \right) \operatorname{Var}(x)
```

```math
\text{Backward: } \operatorname{Var}\!\left(\frac{\partial E}{\partial x}\right) = \left( \prod_{d} n_d^{\text{out}} \operatorname{Var}(w_d) \right) \operatorname{Var}\!\left(\frac{\partial E}{\partial y}\right)
```

A product of many factors → if each factor is > 1 the signal **explodes**, if < 1 it **vanishes** (exponentially in depth). Ideal: each factor ≈ 1.

#### Xavier initialization (linear assumption)

Want $`\prod_d n_d^{\text{in}} \operatorname{Var}(w_d) = C_f`$ and $`\prod_d n_d^{\text{out}} \operatorname{Var}(w_d) = C_b`$ (constants). Satisfied if either:

```math
n_d^{\text{in}} \operatorname{Var}(w_d) = 1
\qquad \text{or} \qquad
n_d^{\text{out}} \operatorname{Var}(w_d) = 1
```

Either one is fine because $`n_d^{\text{out}} = n_{d+1}^{\text{in}}`$, so

```math
\frac{C_b}{C_f} = \frac{n_{\text{last}}^{\text{out}}}{n_{\text{first}}^{\text{in}}} < \infty
```

#### Kaiming / MSRA initialization (ReLU assumption)

ReLU zeros out half of the signal, adding a factor of $`\tfrac{1}{2}`$ per layer:

```math
\prod_{d} \frac{1}{2} \, n_d^{\text{in}} \operatorname{Var}(w_d) = C_f
\qquad
\prod_{d} \frac{1}{2} \, n_d^{\text{out}} \operatorname{Var}(w_d) = C_b
```

Satisfied if:

```math
n_d^{\text{in}} \operatorname{Var}(w_d) = 2
\quad \Longrightarrow \quad
\operatorname{Var}(w_d) = \frac{2}{n_d^{\text{in}}}
```

- With $`D`$ layers, the factor of 2 has an **exponential impact** ($`2^{D}`$) compared with Xavier.
- 22-layer ReLU net: good init **converges faster**.
- 30-layer ReLU net: good init is the difference between converging and **not converging at all**.

### 5.3 Batch Normalization

- Normalizing the input is old advice (LeCun et al. 1998, *Efficient Backprop*).
- **BN:** normalize **each layer**, for **each mini-batch**:

```math
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}},
\qquad
y = \gamma \hat{x} + \beta
```

- Greatly **accelerates** training.
- **Less sensitive** to initialization.
- Improves **regularization**.
- (Ioffe & Szegedy: BN-Inception reaches the baseline Inception accuracy in far fewer iterations.)

### 5.4 The degradation problem

With smart init + BN, is building better networks as simple as **stacking more layers**? **No.**

- Plain nets (stacked 3×3 convs) on **CIFAR-10**: a 56-layer net has **higher training error *and* test error** than a 20-layer net.
- Same on **ImageNet**: plain-34 is worse than plain-18.
- Since the **training** error is higher, this is **not overfitting** — it's an **optimization** problem.

**Argument by construction:** a deeper model has a richer solution space, so it should not have higher training error. Take a learned shallow model, copy its layers, and set the extra layers to **identity** → the deeper model achieves *at least* the same training error. So a solution exists; the solvers just **can't find it**.

### 5.5 Residual learning

Instead of asking a stack of layers to learn the desired mapping $`H(x)`$ directly, let them learn the **residual** $`F(x)`$:

```math
H(x) = F(x) + x
```

```
      x ─────────────────────┐
      │                      │ identity
  [weight layer]             │
      │ relu                 │
  [weight layer]             │
      │  F(x)                │
      └──────── (+) ◄────────┘
                 │ relu
           H(x) = F(x) + x
```

- If identity is optimal, it is easy: push the weights to **0**.
- If the optimal mapping is **close to identity**, it is easier to learn small fluctuations around it.

> ⌚ **Intuition:** fine-tuning a mechanical watch's regulator. You don't rebuild the movement each time; you start from "already about right" (identity) and only make a small adjustment (the residual $`F`$). Learning the small correction is much easier than learning the whole thing from scratch.

#### Network design

- Keep it simple, VGG-style: (almost) all 3×3 convs.
- When spatial size is halved (/2), the number of filters is doubled (×2).
- No hidden FC layers, no dropout.
- All nets trained from scratch, all use BN, standard hyper-parameters & augmentation.

#### Results

- **CIFAR-10:** ResNets of 20 → 32 → 44 → 56 → 110 layers: deeper = **lower** train *and* test error (opposite of plain nets).
- **ImageNet:** ResNet-34 beats ResNet-18 (opposite of plain nets).
- Deep ResNets can be trained **without difficulty**.

| Model (10-crop, top-5 val error %) | Error |
|---|---|
| ResNet-34 | 7.4 |
| ResNet-50 | 6.7 |
| ResNet-101 | 6.1 |
| ResNet-152 | **5.7** |

ResNet-152 still has **lower time complexity** than VGG-16/19.

### 5.6 Why it works: smooth forward / backward propagation

#### Forward

With identity shortcuts (and identity after-add — see Section 6), each block is $`x_{l+1} = x_l + F(x_l)`$. Unrolling:

```math
\begin{aligned}
x_{l+1} &= x_l + F(x_l) \\
x_{l+2} &= x_{l+1} + F(x_{l+1}) = x_l + F(x_l) + F(x_{l+1}) \\
x_L &= x_l + \sum_{i=l}^{L-1} F(x_i)
\end{aligned}
```

- Any $`x_l`$ is **directly forward-propagated** to any $`x_L`$, plus a residual.
- Any $`x_L`$ is an **additive** outcome.
- In contrast, a plain net is **multiplicative**:

```math
x_L = \left( \prod_{i=l}^{L-1} W_i \right) x_l
```

#### Backward

By the chain rule on $`x_L = x_l + \sum F(x_i)`$:

```math
\frac{\partial E}{\partial x_l}
= \frac{\partial E}{\partial x_L} \frac{\partial x_L}{\partial x_l}
= \frac{\partial E}{\partial x_L} \left( 1 + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} F(x_i) \right)
```

- The **"1"** term: any $`\frac{\partial E}{\partial x_L}`$ is **directly back-propagated** to any $`\frac{\partial E}{\partial x_l}`$, plus a residual.
- The gradient is **additive** → **unlikely to vanish** (the residual term would have to be exactly −1).
- In contrast, a plain net is multiplicative:

```math
\frac{\partial E}{\partial x_l} = \left( \prod_{i=l}^{L-1} W_i \right) \frac{\partial E}{\partial x_L}
```

> 🏎️ **Intuition:** the identity path is like a DRS-open straight in F1 — the signal has a clean, unobstructed path through the whole lap, while the residual branches are the corners that add (but never block) contributions.

**Requirements for this to hold:**
- shortcut mapping $`h`$ = identity
- after-add mapping $`f`$ = identity

### 5.7 Bottleneck block

Practical design for going deeper (ResNet-50/101/152), with similar complexity to the basic block:

| Basic block (64-d) | Bottleneck block (256-d) |
|---|---|
| 3×3, 64 → ReLU | 1×1, 64 → ReLU *(reduce)* |
| 3×3, 64 | 3×3, 64 → ReLU |
| + identity → ReLU | 1×1, 256 *(restore)* |
| | + identity → ReLU |

### 5.8 Representation, optimization, generalization

| Issue | Meaning | How ResNet addresses it |
|---|---|---|
| **Representation ability** | Ability to fit training data *if the optimum could be found*. If A's solution space ⊇ B's, A should be better. | No explicit advantage (only a **re-parameterization**), but it **allows models to go deeper**. |
| **Optimization** | Feasibility of *finding* an optimum. Not all models are equally easy to optimize. | Very **smooth forward/backward propagation** → greatly eases optimizing deep models. |
| **Generalization** | Once training data is fit, how good is test performance? | Not addressed explicitly, but **deeper + thinner** yields good generalization. |

**Loss landscape** (Li et al., NeurIPS 2018, *Visualizing the Loss Landscape of Neural Nets*): a 56-layer ConvNet **without** skip connections has a highly chaotic, non-convex loss surface; a 56-layer ResNet **with** skip connections has a much **smoother, near-convex** surface.

---

## 6. Importance of the Identity: from 100 to 1000 layers

> He, Zhang, Ren & Sun, *"Identity Mappings in Deep Residual Networks"*, ECCV 2016.

General residual unit:

```math
x_{l+1} = f\big( h(x_l) + F(x_l) \big)
```

- $`h`$ = **shortcut mapping** (original ResNet: identity)
- $`f`$ = **after-add mapping** (original ResNet: ReLU)

Two sets of experiments, on ResNets with > 100 layers (deeper models suffer more from optimization difficulties):

1. What if the **shortcut** mapping $`h`$ is **not** identity?
2. What if the **after-add** mapping $`f`$ **is** identity?

### 6.1 Set 1 — shortcut variants (ResNet-110, CIFAR-10)

| Variant | Shortcut $`h(x)`$ | Error (%) |
|---|---|---|
| (a) original | $`x`$ | **6.6** |
| (b) constant scaling | $`0.5\,x`$ (residual also ×0.5) | 12.4 |
| (c) exclusive gating | $`\text{gate} \cdot x`$, residual × (1 − gate) *(≈ Highway Net)* | 8.7 |
| (d) shortcut-only gating | $`\text{gate} \cdot x`$ | 12.9 |
| (e) 1×1 conv shortcut | $`\text{conv}(x)`$ | 12.2 |
| (f) dropout shortcut | $`\text{dropout}(x)`$ | > 20 |

**All non-identity shortcuts are worse** — the shortcuts are **blocked by multiplications**.

#### Why multiplicative shortcuts hurt

If $`h(x) = \lambda x`$ (and $`f`$ = identity), unrolling gives:

```math
x_L = \lambda^{L-l} x_l + \sum_{i=l}^{L-1} \hat{F}(x_i)
```

```math
\frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L} \left( \lambda^{L-l} + \frac{\partial}{\partial x_l} \sum_{i=l}^{L-1} \hat{F}(x_i) \right)
```

(The slide writes the exponent as $`L-1`$; the number of blocks between $`l`$ and $`L`$ is $`L-l`$.)

- $`\lambda > 1`$ → explodes; $`\lambda < 1`$ → **decays exponentially**. Shortcuts are blocked; direct propagation is decayed.

#### Gating

- Gating *should* have **better representation ability** (identity is a special case: gate = 1).
- But **optimization difficulty dominates** the results — the gated net has higher training loss than the identity net.

### 6.2 Set 2 — after-add mapping $`f`$

| Design | Order inside the unit | $`f`$ |
|---|---|---|
| Original ResNet | weight → BN → ReLU → weight → BN → **add** → ReLU | ReLU |
| BN after add | weight → BN → ReLU → weight → **add** → BN → ReLU | BN + ReLU |
| **Pre-activation ResNet** | BN → ReLU → weight → BN → ReLU → weight → **add** | **identity** |

- $`f`$ = BN + ReLU: **BN can block propagation** → worse. Keep the shortest path as smooth as possible.
- $`f`$ = ReLU: **ReLU can block propagation when there are 1000 layers**.
- **Pre-activation** (move BN & ReLU *before* the weights, so $`f`$ = identity) eases optimization **and** improves generalization.
- 1001-layer ResNet on CIFAR-10: original 7.61% → pre-activation **4.92%**.

### 6.3 Results

**CIFAR-10 / CIFAR-100** (moderate augmentation):

| Method | CIFAR-10 error (%) | CIFAR-100 error (%) |
|---|---|---|
| NIN | 8.81 | 35.68 |
| DSN | 8.22 | 34.57 |
| FitNet | 8.39 | 35.04 |
| Highway | 7.72 | 32.39 |
| ResNet-110 (1.7M) | 6.61 | — |
| ResNet-164 (1.7M) | — | 25.16 |
| ResNet-1202 (19.4M) | 7.93 | — |
| ResNet-1001 (10.2M) | — | 27.82 |
| ResNet-164, pre-act (1.7M) | 5.46 | 24.33 |
| **ResNet-1001, pre-act (10.2M)** | **4.92** (4.89 ± 0.14) | **22.71** (22.68 ± 0.22) |

**ImageNet** (single-crop 320×320, val error):

| Method | Augmentation | Top-1 (%) | Top-5 (%) |
|---|---|---|---|
| ResNet-152, original | scale | 21.3 | 5.5 |
| ResNet-152, pre-act | scale | 21.1 | 5.5 |
| ResNet-200, original | scale | 21.8 | 6.0 |
| ResNet-200, pre-act | scale | 20.7 | 5.3 |
| **ResNet-200, pre-act** | scale + aspect ratio | **20.1** | **4.8** |

Note: original ResNet-200 is *worse* than ResNet-152; pre-activation fixes this.

---

## 7. Summary

**Modern CNNs**

- AlexNet (2012): ReLU, dropout, augmentation → deep learning wave.
- VGG (2014): only 3×3 conv + 2×2 pool; 138M params, mostly in FC layers.
- GoogLeNet (2014): Inception modules, 1×1 bottlenecks, no FC → only 5M params.

**ResNet**

- Plain deep nets suffer from **degradation** (an optimization problem, not overfitting).
- Learn the residual: $`H(x) = F(x) + x`$.
- Identity shortcuts make forward signals and backward gradients **additive**, so they don't vanish.

**Identity mappings (pre-activation ResNet)**

- Keep the **shortest path as smooth as possible** by making **both** $`h`$ and $`f`$ identity.
- Forward/backward signals flow directly through this path; features of any layer are **additive** outcomes.
- **1000-layer ResNets** can be trained easily and get better accuracy.

```math
\boxed{\;x_L = x_l + \sum_{i=l}^{L-1} F(x_i)
\qquad
\frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L}\left(1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1} F(x_i)\right)\;}
```

---

### References

- L. Gatys, A. Ecker, M. Bethge. *Image Style Transfer Using Convolutional Neural Networks.* CVPR 2016.
- A. Krizhevsky, I. Sutskever, G. Hinton. *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012.
- K. Simonyan, A. Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* 2014.
- C. Szegedy et al. *Going Deeper with Convolutions.* 2014.
- S. Ioffe, C. Szegedy. *Batch Normalization.* 2015.
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- K. He, X. Zhang, S. Ren, J. Sun. *Identity Mappings in Deep Residual Networks.* ECCV 2016.
- H. Li et al. *Visualizing the Loss Landscape of Neural Nets.* NeurIPS 2018.
- Slide credits: Prof. Kosta Derpanis (style transfer); Fei-Fei Li, Andrej Karpathy & Justin Johnson (CS231n); Kaiming He (ResNet tutorial); Brian Kulis.
