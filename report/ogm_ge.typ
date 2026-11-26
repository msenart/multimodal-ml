// Layout of this section: more air around paragraphs, lists and tables.
#set par(spacing: 1.3em)
#set list(spacing: 0.9em)
#show figure: set block(spacing: 2.2em)
#show figure.caption: set text(size: 9.5pt)
#set table(inset: (x: 7pt, y: 5pt))
#show table: set text(size: 9.5pt)

// A difference with its 95% CI on a second, smaller line.
#let ci(m, lo, hi, strong: false) = {
  let v = if strong { [*#m*] } else { [#m] }
  stack(spacing: 3pt, v, text(size: 8pt, fill: luma(90))[[#lo, #hi]])
}

= Applying OGM-GE to the concatenation fusion <sec-ogm-ge>

We apply OGM-GE @peng2022ogm to the concatenation fusion of the previous section and test whether it helps on CMU-MOSI. Every part of the method can be switched on and off from the command line, and `--ogm none` gives exactly the training of the previous section.

== Method

*OGM-GE in the paper.* At each step, the paper scores each modality $m$: $s_m$ is the softmax probability that its part of the logits gives to the true class, summed over the batch. The ratio $rho_m$ of the two scores tells which modality dominates, and the gradient of the dominant encoder is multiplied by
$ k_m = 1 - tanh(alpha (rho_m - 1)) quad "if" rho_m > 1, quad k_m = 1 "otherwise". $
This is OM (_on-the-fly gradient modulation_). GE (_generalization enhancement_) then adds a Gaussian noise with the standard deviation of the gradient.

*Our adaptation.* We have three modalities and a regression. With $c_m = W_m phi_m + b\/3$ (@eq-concat):
$ hat(y)_m = c_m + sum_(j != m) overline(c)_j, quad s_m = sum_(i in "batch") exp(-(y_i - hat(y)_(m,i))^2 \/ 2), quad rho_m = s_m / (max_(j != m) s_j). $

The choices, and why:
- *Gaussian likelihood as score.* The softmax probability of the true class is the likelihood of the label; for a regression trained with the MSE, the matching model is a Gaussian. The score is always positive, so $rho_m$ is defined.
- *Other modalities held at their batch mean* ($overline(c)_j$). $c_m$ alone is not a prediction (the split of $b$ is arbitrary and the other parts have a non-zero mean); $hat(y)_m$ is the prediction of $m$ alone.
- *Ratio to the best other modality.* With two modalities this is the paper's ratio. With the _mean_ of the others, a useless modality lowers the mean and all the others are slowed down. We tried it first; it leads to the same conclusions.
- *Only the encoders are modulated*, and *GE noise only on weight matrices*, as in the paper and its official code.
- *$k_m$ recomputed at every batch, during all epochs*, as in the paper. Our runs are short (at most 20 epochs).

*Options* (each one can be switched off):
- `--ogm none|ogm|ogm-ge` (default `none`): plain training, OM alone, or OM + GE;
- `--ogm-alpha` (default 0.1; $alpha = 0$ gives $k = 1$) and `--ogm-ratio max|mean`;
- `--ogm-start`, `--ogm-end`: epochs during which OGM is applied;
- `--optimizer adam|sgd` (default `adam`), `--momentum`, `--weight-decay`.

== Protocol

- *Same model and training* as the previous section: 3 layers of 256 units, MSE, batches of 16, at most 20 epochs, early stopping on the validation MAE.
- *10 paired seeds.* A baseline and an OGM run with the same seed share the initialization and the batch order, so we compare them seed by seed.
- *Two optimizers.* Adam ($10^(-3)$, as before) and SGD with momentum 0.9, the optimizer of the paper. The SGD learning rate ($3 dot 10^(-3)$) is chosen for the baseline on the validation MAE.
- *$alpha in {0.1, 0.3, 1, 3}$*, for OM and OGM-GE. $alpha$ is selected on the validation MAE, never on the test set.
- *Statistics.* Per seed, the difference OGM − baseline; we report its mean and a 95% bootstrap confidence interval. Primary metric: test MAE, tested with a Wilcoxon signed-rank test, Holm-corrected over the 8 conditions ($p_"Holm"$). We also give the _minimum detectable effect_ (MDE) of a paired t-test with 10 seeds (power 80%).

In total: 580 runs.

== Checking the implementation

*Unit tests* (16, no dataset needed) check that:
- with `--ogm none`, training is bit-identical to the default routine, and with $alpha = 0$ and no GE, OGM gives exactly the baseline;
- $s_m$, $rho_m$ and $k_m$ match hand-computed values;
- only the encoder of the modulated modality is affected, by exactly $k_m$; the GE noise has zero mean and the standard deviation of the gradient.

*Synthetic control.* On synthetic data, the "text" is easy (linear) but has many noise dimensions to memorize, the "vision" carries information through a non-linear code (slower to learn), and the "audio" is pure noise. In the _independent_ scenario, the label is $s_T + 0.7 s_V$ and the fusion needs both. In the _redundant_ scenario, both see the same signal and vision is the better modality. Both were fixed before the runs.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Scenario*], [*Opt.*], [*Baseline MAE*], [*OM: Δ MAE*], [*OGM-GE: Δ MAE*]),
    table.hline(stroke: 0.5pt),
    [Independent], [Adam], [81.10 ± 1.83], ci("−0.03", "−0.23", "+0.14"), ci("−1.51", "−2.01", "−0.95", strong: true),
    [Independent], [SGD],  [74.60 ± 2.44], ci("+0.04", "−0.02", "+0.09"), ci("−0.18", "−0.57", "+0.23"),
    [Redundant],   [Adam], [53.99 ± 4.72], ci("+0.19", "+0.01", "+0.46"), ci("−0.00", "−0.84", "+0.97"),
    [Redundant],   [SGD],  [52.59 ± 3.79], ci("+0.13", "−0.02", "+0.29"), ci("+0.74", "−0.54", "+2.08"),
    table.hline(),
  ),
  caption: [Synthetic control, $alpha = 0.3$ (selected on validation). OGM − baseline (× 100), mean and 95% CI over 10 seeds. Bold: significant after Holm correction ($p_"Holm" = 0.02$).],
) <tab-ogm-synth>

The modulation does what it should: the text, which learns first, gets $k < 1$ ($k = 0.76$ with Adam, $alpha = 1$), and the other modalities keep $k approx 1$. But OM brings no gain (@tab-ogm-synth). The only gain (independent, Adam, +2.0 Corr) appears with GE and only with Adam: it comes from the gradient noise acting as a regularizer, not from the rebalancing.

Here the joint training does not under-train vision: a linear probe of the vision encoder gives the same correlation trained jointly (50.8), alone (51.2), and *never trained* (49.3). With low-dimensional inputs, random features are already enough for a linear probe, so *a probe must always be compared with a random encoder*.

== Adam cancels most of the modulation <sec-ogm-adam>

Adam updates a parameter by $Delta theta = -eta thin hat(m) \/ (sqrt(hat(v)) + epsilon)$, where $hat(m)$ and $hat(v)$ are running means of the gradient and of its square. If all the gradients of an encoder are multiplied by the same $k$, both $hat(m)$ and $sqrt(hat(v))$ are multiplied by $k$, and the update does not change: only the _variations_ of $k$ matter. SGD has no such normalization, and the paper uses SGD.

We measure this on MOSI: the distance travelled by the text encoder's parameters during the first 5 epochs, with OM divided by without OM (same seed).

#figure(
  table(
    columns: 5,
    align: center,
    stroke: none,
    table.hline(),
    table.header([], table.cell(colspan: 2)[*Adam*], table.cell(colspan: 2)[*SGD*]),
    table.hline(start: 1, end: 3, stroke: 0.3pt),
    table.hline(start: 3, end: 5, stroke: 0.3pt),
    [*$alpha$*], [mean $k_T$], [distance ratio], [mean $k_T$], [distance ratio],
    table.hline(stroke: 0.5pt),
    [0.1], [0.96], [1.00], [0.97], [0.98],
    [0.3], [0.88], [0.99], [0.91], [0.93],
    [1],   [0.65], [0.95], [0.74], [0.78],
    [3],   [0.35], [0.89], [0.50], [0.56],
    table.hline(),
  ),
  caption: [Text encoder on CMU-MOSI, OM only, first 5 epochs, mean over 10 seeds.],
) <tab-ogm-adam>

With SGD, the text encoder is slowed down by about $k$. With Adam and $alpha = 3$, OGM asks for a 65% slowdown and gets 11%.

== Results on CMU-MOSI

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([], [], table.cell(colspan: 2)[*Adam*], table.cell(colspan: 2)[*SGD*]),
    table.hline(start: 2, end: 4, stroke: 0.3pt),
    table.hline(start: 4, end: 6, stroke: 0.3pt),
    [*Method*], [*$alpha$*], [Δ MAE], [Δ Corr], [Δ MAE], [Δ Corr],
    table.hline(stroke: 0.5pt),
    [Baseline], [], [94.51 ± 1.86], [67.31 ± 0.96], [94.93 ± 2.27], [66.68 ± 1.39],
    table.hline(stroke: 0.3pt),
    [OM], [0.1 †], ci("+0.10", "−1.76", "+1.96"), ci("+0.02", "−0.74", "+0.79"), ci("−0.34", "−2.06", "+1.22"), ci("+0.84", "+0.21", "+1.50"),
    [],   [0.3],   ci("+1.27", "−0.67", "+3.96"), ci("−0.24", "−0.81", "+0.32"), ci("−0.22", "−0.98", "+0.41"), ci("+0.59", "+0.04", "+1.29"),
    [],   [1],     ci("+1.88", "−0.10", "+4.23"), ci("−0.47", "−1.20", "+0.17"), ci("+1.11", "−0.39", "+2.82"), ci("+0.81", "+0.24", "+1.32"),
    [],   [3],     ci("+1.62", "−1.02", "+4.61"), ci("−0.64", "−1.53", "+0.37"), ci("+1.92", "−0.60", "+4.60"), ci("+0.73", "−0.73", "+2.09"),
    table.hline(stroke: 0.3pt),
    [OGM-GE], [0.1 †], ci("+0.70", "−1.74", "+3.34"), ci("+0.45", "−0.12", "+1.08"), ci("−0.42", "−1.59", "+0.93"), ci("+0.82", "+0.21", "+1.47"),
    [],       [0.3],   ci("+0.41", "−1.84", "+2.53"), ci("+0.12", "−1.01", "+1.14"), ci("−0.28", "−1.27", "+0.87"), ci("+0.45", "+0.00", "+0.94"),
    [],       [1],     ci("+1.05", "−0.95", "+2.86"), ci("−0.02", "−1.00", "+0.82"), ci("+1.28", "−0.13", "+2.91"), ci("+0.27", "−0.78", "+1.11"),
    [],       [3],     ci("+1.77", "−0.82", "+4.49"), ci("+0.14", "−0.83", "+0.95"), ci("+2.17", "+0.56", "+3.72"), ci("+1.02", "+0.17", "+1.78"),
    table.hline(),
  ),
  caption: [OGM − baseline on the CMU-MOSI test set (× 100; negative Δ MAE is better). First row: absolute baseline. Mean and 95% CI over 10 paired seeds. †: $alpha$ selected on validation.],
) <tab-ogm-mosi>

*No condition improves the MAE*, the primary metric: all 16 comparisons have $p_"Holm" >= 0.29$, and the selected $alpha = 0.1$ gives a difference close to zero. Strong modulation ($alpha >= 1$) tends to hurt. The MDE is 1.2 to 4.8 MAE points, so a real gain of about 1 point could be missed.

With SGD, the correlation rises by about 0.8 points. This secondary metric is not corrected for multiple tests, and it only brings SGD to the level of the Adam baseline (67.5 against 67.3).

== Does OGM-GE rebalance the fusion?

*The text still dominates*: it provides 93–97% of the prediction in every condition (96% without OGM).

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Encoder*], [*Random*], [*Alone*], [*Joint*], [*+ OM*], [*+ OGM-GE*]),
    table.hline(stroke: 0.5pt),
    [Text],  [47.8 ± 2.5], [66.5 ± 0.7], [67.1 ± 0.8], [67.1 ± 1.2], [67.3 ± 0.8],
    [Audio], [24.6 ± 1.1], [23.6 ± 1.2], [23.7 ± 0.8], [23.2 ± 0.9], [24.2 ± 1.0],
    [Video], [12.5 ± 4.6], [12.8 ± 3.1], [14.1 ± 3.2], [14.9 ± 3.1], [14.7 ± 3.7],
    table.hline(),
  ),
  caption: [Linear probe of each encoder on the CMU-MOSI test set (Corr × 100, 10 seeds, Adam, $alpha = 0.1$). Random: initial weights, never trained.],
) <tab-ogm-probe10>

*The audio and video encoders learn nothing useful*, alone or jointly: a linear layer on _random_ features does as well (@tab-ogm-probe10). Only the text encoder learns (47.8 → 66.5). With 10 seeds, the drop of the jointly trained video encoder seen with 3 seeds (@tab-ogm-probe) disappears (14.1 jointly, 12.8 alone). There is no under-trained encoder for OGM-GE to fix.

*How much can audio and video add?* A linear regression on the predictions of the three uni-modal models, fitted on validation, improves the MAE over the text alone by 2.0 points [1.4, 2.5], but the correlation by only 0.11 [−0.20, +0.41]. Our fusion already reaches this level. The room for improvement is very small.

== Conclusion

- On CMU-MOSI, OGM-GE brings *no significant gain* to the concatenation fusion, with Adam or SGD, for any $alpha$.
- The cause is not the optimization imbalance targeted by the paper: the audio and video features carry very little information, and even an optimal combination of uni-modal models barely beats the text.
- *With Adam, the modulation is mostly cancelled*; OGM-GE with Adam mostly tests the GE noise.
- *Limits*: one dataset, small encoders on pre-extracted features, effects below the MDE (1 to 5 MAE points) cannot be excluded.
