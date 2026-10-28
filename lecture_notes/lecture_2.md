# Probability & ML Basics — Study Note

*EC523 · Lecture: Probability and Math Basics (Kayhan Batmanghelich)*

> Math uses GitHub's ` ```math ` blocks for display equations and `` $`...`$ `` for inline math. Both render natively on github.com.

---

## 1. Expectation, variance, covariance

Expectation is the probability-weighted average of a random variable; variance measures spread around it; covariance measures how two variables move together.

```math
E[V] = \int_X x \, p(x) \, dx \qquad E[V] = \sum_x x \, p(x) \ \text{(discrete)}
```

```math
\mathrm{Var}(V) = E\big[(V - E[V])^2\big] = E[V^2] - (E[V])^2 \qquad \sigma^2 = \mathrm{Var}(V) \qquad \mu = E[V]
```

```math
\mathrm{Cov}(V, W) = E\big[(V - E[V])(W - E[W])\big]
```

**Properties to memorize**

```math
E[V + W] = E[V] + E[W] \qquad E[cV] = c \, E[V]
```

```math
\mathrm{Var}(V + W) = \mathrm{Var}(V) + \mathrm{Var}(W) + 2 \, \mathrm{Cov}(V, W) \qquad \mathrm{Var}(cV) = c^2 \, \mathrm{Var}(V)
```

Linearity of expectation holds even when $`V`$ and $`W`$ are dependent. The variance rule needs the covariance term unless $`V`$ and $`W`$ are uncorrelated.

**Law of large numbers.** If $`V_1, \dots, V_n`$ are i.i.d. with mean $`E[V]`$, the sample average converges to $`E[V]`$:

```math
\lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^{n} V_i = E[V]
```

**Example (F1 lap times).** A driver's lap time $`V`$ in a stint has mean 92.0 s and standard deviation 0.3 s. Tyre wear $`W`$ adds a delay with mean 0.4 s and standard deviation 0.2 s, and $`\mathrm{Cov}(V, W) = 0.02 \ \text{s}^2`$. Then

```math
E[V + W] = 92.4 \ \text{s} \qquad \mathrm{Var}(V + W) = 0.09 + 0.04 + 2 \cdot 0.02 = 0.17 \ \text{s}^2
```

so the spread is about 0.41 s. Averaging many laps, the team's measured mean converges to the true 92.4 s — the law of large numbers at work on a race engineer's timing sheet.

---

## 2. Markov's inequality and the union bound

Markov's inequality bounds how often a nonnegative variable can be large using only its mean; the slides call it possibly the most important probability fact in ML.

```math
V \ge 0 \implies P(V > z) \le \frac{E[V]}{z} \quad \text{for any } z > 0
```

**Proof.** On the region $`x > z`$ we have $`x / z > 1`$, so weighting by $`x / z`$ can only increase the integral; then extend the range down to 0.

```math
\begin{aligned}
P(V > z) &= \int_z^{\infty} p(x) \, dx \\
&\le \frac{1}{z} \int_z^{\infty} x \, p(x) \, dx \\
&\le \frac{1}{z} \int_0^{\infty} x \, p(x) \, dx = \frac{E[V]}{z}
\end{aligned}
```

**Applied to variance (Chebyshev).** $`(V - E[V])^2`$ is always nonnegative and its mean is $`\mathrm{Var}(V)`$, so

```math
P\big((V - E[V])^2 > z\big) \le \frac{\mathrm{Var}(V)}{z}
```

**Union bound.** For any events, $`P(A \text{ or } B) \le P(A) + P(B)`$. If $`A_1, \dots, A_N`$ are "bad" events, each with probability at most $`\delta`$:

```math
P\Big(\bigcup_{i=1}^{N} A_i\Big) \le N \delta \quad \implies \quad P(\text{none of the } A_i) \ge 1 - N \delta
```

**Example (watch rate).** An automatic movement's absolute daily rate error $`V`$ has mean 4 s/day. Markov gives $`P(V > 20) \le 4 / 20 = 0.2`$, with no knowledge of the distribution. If the variance of the signed rate is 9 (s/day)², Chebyshev gives $`P(\text{deviation}^2 > 100) \le 9 / 100`$, so the rate strays more than 10 s/day from its mean at most 9% of the time.

For the union bound: if each of 5 complications (date, moonphase, chronograph, power reserve, GMT) fails a test with probability at most 0.01, the watch passes all five with probability at least $`1 - 5 \cdot 0.01 = 0.95`$, even if failures are correlated.

---

## 3. Common densities and the central limit theorem

Three distributions recur all course: the Gaussian (continuous values), its multivariate form (vectors), and the Poisson (counts).

**Gaussian $`\mathcal{N}(\mu, \sigma^2)`$.** The normalizer is $`\sqrt{2 \pi \sigma^2}`$ (the slide image drops the square on $`\sigma`$).

```math
p(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)
```

Rule of thumb: about 68% of the mass lies within $`\mu \pm \sigma`$, 95% within $`\mu \pm 2\sigma`$, 99.7% within $`\mu \pm 3\sigma`$.

**Multivariate Gaussian.** $`\mu`$ is a $`d`$-vector and $`\Sigma`$ a symmetric positive-definite covariance matrix.

```math
p(x; \mu, \Sigma) = \frac{1}{(2 \pi)^{d/2} \sqrt{\det(\Sigma)}} \exp\left(-\frac{1}{2} (x - \mu)^{T} \Sigma^{-1} (x - \mu)\right)
```

The slide's question — a linear combination of Gaussian variables is Gaussian, with what mean and variance?

```math
x \sim \mathcal{N}(\mu, \Sigma) \implies a^{T} x \sim \mathcal{N}\left(a^{T} \mu, \ a^{T} \Sigma a\right) \qquad A x + b \sim \mathcal{N}\left(A \mu + b, \ A \Sigma A^{T}\right)
```

**Poisson($`\lambda`$).** A distribution over counts $`x = 0, 1, 2, \dots`$ (zero included), with mean = variance = $`\lambda`$.

```math
p(x; \lambda) = \frac{\lambda^{x} e^{-\lambda}}{x!}
```

**Central limit theorem.** If $`V_1, \dots, V_N`$ are independent (not necessarily identically distributed, under mild conditions) and $`S`$ is their average, then $`S`$ is approximately Gaussian with $`\mu = E[S]`$ and $`\sigma^2 = \mathrm{Var}(S)`$. In the i.i.d. case with variance $`\sigma_V^2`$:

```math
S = \frac{1}{N} \sum_{i=1}^{N} V_i \ \approx \ \mathcal{N}\left(E[V], \ \frac{\sigma_V^2}{N}\right)
```

**Example (soccer goals).** Goals in a match are well modelled as Poisson with $`\lambda \approx 2.7`$.

```math
P(0\text{–}0) = e^{-2.7} \approx 0.067 \qquad P(\text{exactly 3 goals}) = \frac{2.7^3 \, e^{-2.7}}{3!} \approx 0.22
```

Each match's count is skewed and discrete, yet a season average over 38 matches is close to $`\mathcal{N}(2.7, \ 2.7 / 38)`$, with a standard deviation of about 0.27 goals. That is the CLT turning a lumpy count into a bell curve.

---

## 4. The ML workflow and linear classifiers

Use ML when you cannot write the answer down but do have input/output examples; don't use it for quadratic equations or matrix rank.

**Typical flow**

1. Gather a training set $`S_{\text{train}} = \{(x_1, y_1), \dots, (x_N, y_N)\}`$.
2. Choose a hypothesis class $`H`$, a set of functions $`f: X \to Y`$.
3. Run a training algorithm to pick $`\hat{f} \in H`$ with $`\hat{f}(x) \approx y`$.
4. Evaluate $`\hat{f}`$ on a new test set $`S_{\text{test}}`$.

**Parametrized hypothesis class.** Fix one model $`f(x, \theta)`$ and let the parameters range over a parameter space $`\Theta`$; training means finding the parameter estimate $`\hat{\theta}`$.

```math
H = \{\, x \mapsto f(x, \theta) \ : \ \theta \in \Theta \,\} \qquad \hat{f}(x) = f(x, \hat{\theta})
```

**Linear classifier.** For $`x \in \mathbb{R}^d`$ and $`y \in \{-1, +1\}`$, the decision boundary is the hyperplane $`\theta \cdot x = 0`$. Each feature is an "expert" voting with weight $`\theta_j`$; the sign is a weighted majority rule.

```math
\hat{y} = \mathrm{sign}(\theta \cdot x) = \mathrm{sign}(\theta_1 x_1 + \cdots + \theta_d x_d)
```

**Multiclass (CIFAR-10).** A 32 × 32 × 3 image flattens to $`x \in \mathbb{R}^{3072}`$. $`W`$ is 10 × 3072 and $`b`$ is 10 × 1, giving 10 class scores; the prediction is the class with the largest score. Each row of $`W`$ acts as a learned template for one class, and each row also defines one hyperplane in pixel space.

```math
f(x_i; W, b) = W x_i + b \qquad \hat{y}_i = \arg\max_k \, \big(W x_i + b\big)_k
```

**Perceptron (mistake-driven estimation).** Only when the prediction is wrong, nudge $`\theta`$ toward the correct side. Its average error drops as more labelled examples arrive.

```math
\text{if } \hat{y} \ne y: \quad \theta_j \leftarrow \theta_j + y \, x_j \qquad j = 1, \dots, d
```

**Example (guitar takes).** Label each recorded take +1 (keeper) or −1 (redo) from binary features $`x = [\text{clean tremolo}, \ \text{no fret buzz}, \ \text{audible string squeak}]`$. Start at $`\theta = [0, 0, 0]`$.

- Take $`x = [1, 1, 0]`$, $`y = +1`$: the score is 0, treated as wrong, so $`\theta`$ becomes $`[1, 1, 0]`$.
- Take $`x = [0, 1, 1]`$, $`y = -1`$: the score is 1, predicted +1, wrong, so $`\theta`$ becomes $`[1, 0, -1]`$.

Now squeak votes against a keeper, clean tremolo votes for it, and fret buzz is neutral until more data arrives.

---

## 5. Loss functions

A loss function quantifies disagreement between predicted scores and ground-truth labels; training minimizes its average.

**Natural choices.** Classification: the 0-1 loss, whose average $`L_S(\theta)`$ is the fraction of mistakes on $`S`$. Regression: the squared loss.

```math
\ell_{0\text{-}1}(x, y, \theta) = \begin{cases} 0 & f(x, \theta) = y \\ 1 & \text{otherwise} \end{cases} \qquad \ell_{\text{sq}}(x, y, \theta) = \lVert f(x, \theta) - y \rVert^2
```

**Hinge loss.** $`w \cdot x_i + b`$ is the scoring function, $`y_i \in \{-1, +1\}`$ is the label, and the 1 is the margin: a point must be correct *by at least 1* to cost nothing.

```math
L_i = \max\big(0, \ 1 - y_i \, (w \cdot x_i + b)\big)
```

**Logistic loss.** A smooth version that never reaches exactly zero.

```math
L_i = \ln\big(1 + \exp(-y_i \, f(x_i))\big)
```

All three are functions of the margin $`m = y_i f(x_i)`$: positive means correct, larger means more confident. The 0-1 loss is a step (flat almost everywhere, so its gradient is useless); hinge and logistic are convex upper bounds on it that gradient methods can minimize.

| Margin m | 0-1 loss | Hinge | Logistic (ln) |
| --- | --- | --- | --- |
| −1 (wrong) | 1 | 2 | 1.31 |
| 0 (on boundary) | 0 | 1 | 0.69 |
| 0.5 (barely right) | 0 | 0.5 | 0.47 |
| 1 (at margin) | 0 | 0 | 0.31 |
| 2 (confident) | 0 | 0 | 0.13 |

The slide's plot shows the logistic curve passing through 1 at $`m = 0`$, which is the base-2 log; with $`\ln`$ the value is 0.69. Either way it sits above the 0-1 step.

**Example (rally surface).** A classifier calls a stage gravel (+1) or tarmac (−1) from onboard telemetry. For a gravel stage scoring 0.5, the 0-1 loss is 0 (correct), but hinge still charges 0.5 because the call is not confident enough — pushing the model to separate the surfaces with room to spare.

---

## 6. Train, validate, test — and the i.i.d. assumption

Train on $`S_{\text{train}}`$, tune on $`S_{\text{validate}}`$, and touch $`S_{\text{test}}`$ once at the very end.

- **$`S_{\text{train}}`$** fits $`\hat{\theta}`$ for a given model $`f(x, \theta)`$ and parameter space $`\Theta`$.
- **$`S_{\text{validate}}`$** scores each candidate; if performance is bad, change $`H`$ or the training algorithm and retrain.
- **$`S_{\text{test}}`$** gives the final, unbiased estimate. In practice people often reuse the validation set as the test set, which makes the reported number optimistic.

**K-fold cross-validation (K = 5).** Hold out the test set first. Split the rest into 5 folds; in each round train on 4 folds and validate on the fifth, rotating so every fold validates once. Average the 5 validation scores to pick the model and hyperparameters, then evaluate once on the untouched test set.

**I.i.d. assumption.** Almost all ML analysis assumes every $`(x, y)`$ pair in train and test is drawn independently from the same distribution. It is often violated but is "good enough" for many settings. This course is mostly about empirically successful choices of the model $`f(x, \theta)`$.

**Example (soccer xG model).** Fitting an expected-goals model on 2019–2024 shots and testing on the 2025 season breaks i.i.d. if rule changes or new tactics shift the shot distribution. Random K-fold across seasons would hide that drift; a season-wise split exposes it.

---

## 7. Maximum likelihood estimation

MLE picks the $`\theta`$ that makes the observed training labels most probable; with Gaussian noise it reduces exactly to least squares.

**Setup.** The hypothesis class is a set of conditional distributions $`H = \{p(y \mid x; \theta) : \theta \in \Theta\}`$. Assuming the pairs are i.i.d., the likelihood factorizes, and we maximize its log because sums are easier to differentiate and numerically stabler than long products of small numbers.

```math
P\big[(x_1, y_1), \dots, (x_N, y_N) \mid x_1, \dots, x_N\big] = \prod_{i=1}^{N} p(y_i \mid x_i; \theta)
```

```math
\mathcal{L}(\theta) = \sum_{i=1}^{N} \log p(y_i \mid x_i; \theta) \qquad \hat{\theta} = \arg\max_{\theta} \mathcal{L}(\theta)
```

**Statistical view of regression.** Observed output = function + noise. Anything the function family can't capture is treated as noise, and $`f(x; w)`$ models the conditional mean.

```math
y = f(x; w) + \varepsilon \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2) \qquad E[y \mid x] = f(x; w)
```

```math
p(y \mid x, w, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(y - f(x; w))^2}{2 \sigma^2}\right)
```

**Log-likelihood → squared loss.** Only the first term depends on $`w`$, so maximizing over $`w`$ is minimizing the sum of squared errors.

```math
\log L(D; w, \sigma^2) = -\frac{1}{2 \sigma^2} \sum_{i=1}^{n} \big(y_i - f(x_i; w)\big)^2 - \frac{n}{2} \log(2 \pi \sigma^2)
```

In general the negative log-probability is the **log-loss**:

```math
\mathrm{Loss}\big(y_i, f(x_i; w)\big) = -\log P(y_i \mid x_i, w, \sigma^2)
```

Swapping the Gaussian for the Laplace density plotted on slide 43 turns the squared error into an absolute error.

**MLE of the noise variance.** Set the derivative with respect to $`\sigma^2`$ to zero at $`\hat{w}`$:

```math
\frac{\partial}{\partial \sigma^2} \log L = -\frac{n}{2 \sigma^2} + \frac{1}{2 \sigma^4} \sum_{i=1}^{n} \big(y_i - f(x_i; \hat{w})\big)^2 = 0 \implies \hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^{n} \big(y_i - f(x_i; \hat{w})\big)^2
```

That is the mean squared prediction error. The slide-49 line is a bad fit by this criterion: it misses the trend, so the residuals are large, every Gaussian factor is small, and the likelihood is low.

**Example (F1 fuel effect).** Model lap time as $`y = w_0 + w_1 x + \varepsilon`$, with $`x`$ = fuel load in kg. Three laps: (100 kg, 93.0 s), (60 kg, 91.9 s), (20 kg, 90.6 s). Least squares gives

```math
\hat{w}_1 = \frac{96}{3200} = 0.03 \ \text{s/kg} \qquad \hat{w}_0 \approx 90.03 \ \text{s}
```

i.e. 0.3 s per 10 kg. Residuals are −0.033, +0.067 and −0.033 s, so $`\hat{\sigma}^2 \approx 0.0022 \ \text{s}^2`$ and $`\hat{\sigma} \approx 0.047 \ \text{s}`$: the driver's lap-to-lap noise once fuel burn is accounted for.

---

## 8. Empirical risk minimization and MAP

ERM — minimize the average training loss — is the most popular and flexible way to train; MLE is the special case where the loss is the log-loss, and MAP adds a prior that acts like a regularizer.

**ERM.** Choose a cost or risk function $`\ell(x, y, \theta)`$, average it over a set $`S`$, and minimize on the training set ("training").

```math
L_S(\theta) = \frac{1}{|S|} \sum_{(x, y) \in S} \ell(x, y, \theta) \qquad \hat{\theta} = \arg\min_{\theta} L_{S_{\text{train}}}(\theta)
```

With $`\ell = -\log p(y \mid x; \theta)`$, minimizing $`L_S`$ is maximizing the log-likelihood, so MLE is a special case of ERM. The bet behind ERM is that $`L_{S_{\text{train}}}(\theta) \approx L_{S_{\text{test}}}(\theta)`$ across $`\theta`$, so the training minimum is also near the test minimum; the validation set checks that bet.

**MAP.** Same hypothesis class of distributions, but now assume nature first drew $`\theta`$ from a known prior $`\pi(\theta)`$, then generated the data from $`p(y \mid x; \theta)`$. Compute the posterior $`\pi(\theta \mid S_{\text{train}})`$ and take its mode, the most likely $`\theta`$. Via Bayes' rule (the evidence term does not depend on $`\theta`$), this becomes — the homework asks you to show it:

```math
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \ \big[\mathcal{L}(\theta) + \log \pi(\theta)\big]
```

$`\log \pi(\theta)`$ plays the role of a regularizer added to an ERM loss. For instance, a Gaussian prior gives L2 weight decay:

```math
\theta \sim \mathcal{N}(0, \tau^2 I) \implies \log \pi(\theta) = -\frac{\lVert \theta \rVert^2}{2 \tau^2} + \text{const}
```

**Example (striker's scoring rate).** A new signing scores 3 goals in his first 2 matches. Treat goals per match as Poisson($`\lambda`$). MLE says $`\hat{\lambda} = 3 / 2 = 1.5`$ goals per match — implausibly high. With a Gamma($`\alpha = 2`$, $`\beta = 4`$) prior (mean 0.5, typical for strikers):

```math
\hat{\lambda}_{\text{MAP}} = \frac{\sum_i x_i + \alpha - 1}{n + \beta} = \frac{3 + 1}{2 + 4} \approx 0.67
```

The prior pulls the estimate toward realistic values; as matches accumulate, the data dominate and MAP approaches MLE.

---

## 9. Self-check questions

- [ ] Why does $`\mathrm{Var}(V + W)`$ need a covariance term, but $`E[V + W]`$ never does?
- [ ] Where exactly is nonnegativity used in the Markov proof?
- [ ] Derive Chebyshev from Markov in one line.
- [ ] If $`x \sim \mathcal{N}(\mu, \Sigma)`$, what are the mean and variance of $`a^{T} x`$?
- [ ] Why is the Poisson mean equal to its variance, and what does that imply for overdispersed count data?
- [ ] Write $`W`$'s shape for CIFAR-10 and explain why each row is a "template".
- [ ] Run one perceptron update by hand on a misclassified point.
- [ ] Compute hinge and logistic loss at margin 0.5; why is hinge zero at margin 1 but logistic is not?
- [ ] Why is the 0-1 loss unusable with gradient descent?
- [ ] Why should the test set be used only once?
- [ ] Show that Gaussian-noise MLE equals least squares, and derive $`\hat{\sigma}^2`$.
- [ ] Which prior turns MAP into L2-regularized least squares?
