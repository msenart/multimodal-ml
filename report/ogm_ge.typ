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

- *Same model and training* as the previous section: 3 layers of 256 units, MSE, batches of 16, at most 20 epochs, early stopping on the validation set.
- *10 paired seeds.* A baseline and an OGM run with the same seed share the initialization and the batch order, so we compare them seed by seed.
- *Two optimizers.* Adam ($10^(-3)$, as before) and SGD with momentum 0.9, the optimizer of the paper. The SGD learning rate ($3 dot 10^(-3)$) is chosen for the baseline on the validation set.
- *$alpha in {0.1, 0.3, 1, 3}$*, for OM and OGM-GE. $alpha$ is selected on the validation set (early-stopping criterion), never on the test set.
- *Metrics*: Acc-2 (_Has0_) and weighted F1, as in the rest of the report.
- *Statistics.* Per seed, the difference OGM − baseline; we report its mean and a 95% bootstrap confidence interval. Each metric is tested with a Wilcoxon signed-rank test, Holm-corrected over the 8 conditions of an optimizer ($p_"Holm"$). We also give the _minimum detectable effect_ (MDE) of a paired t-test with 10 seeds (power 80%).

In total: 580 runs.

== Checking the implementation

*Unit tests* (16, no dataset needed) check that:
- with `--ogm none`, training is bit-identical to the default routine, and with $alpha = 0$ and no GE, OGM gives exactly the baseline;
- $s_m$, $rho_m$ and $k_m$ match hand-computed values;
- only the encoder of the modulated modality is affected, by exactly $k_m$; the GE noise has zero mean and the standard deviation of the gradient.

*Synthetic control.* On synthetic data, the "text" is easy (linear) but has many noise dimensions to memorize, the "vision" carries information through a non-linear code (slower to learn), and the "audio" is pure noise. In the _independent_ scenario, the label is $s_T + 0.7 s_V$ and the fusion needs both. In the _redundant_ scenario, both see the same signal and vision is the better modality. Both were fixed before the runs.

#figure(
  table(
    columns: 7,
    align: (left, left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([], [], [], table.cell(colspan: 2)[*OM*], table.cell(colspan: 2)[*OGM-GE*]),
    table.hline(start: 3, end: 5, stroke: 0.3pt),
    table.hline(start: 5, end: 7, stroke: 0.3pt),
    [*Scenario*], [*Opt.*], [*Baseline Acc-2*], [Δ Acc-2], [Δ F1], [Δ Acc-2], [Δ F1],
    table.hline(stroke: 0.5pt),
    [Independent], [Adam], [70.70 ± 0.98], ci("+0.21", "−0.06", "+0.49"), ci("+0.22", "−0.06", "+0.51"), ci("+1.13", "+0.58", "+1.60"), ci("+1.13", "+0.57", "+1.61"),
    [Independent], [SGD],  [74.44 ± 1.62], ci("−0.07", "−0.22", "+0.07"), ci("−0.07", "−0.21", "+0.07"), ci("−0.15", "−0.44", "+0.14"), ci("−0.15", "−0.44", "+0.14"),
    [Redundant],   [Adam], [84.58 ± 2.11], ci("+0.07", "−0.02", "+0.21"), ci("+0.08", "−0.02", "+0.21"), ci("−0.04", "−0.63", "+0.69"), ci("−0.02", "−0.63", "+0.75"),
    [Redundant],   [SGD],  [84.80 ± 1.57], ci("+0.10", "−0.04", "+0.28"), ci("+0.11", "−0.04", "+0.29"), ci("+0.01", "−0.97", "+1.11"), ci("+0.04", "−0.93", "+1.14"),
    table.hline(),
  ),
  caption: [Synthetic control, $alpha = 0.3$ (selected on validation). OGM − baseline (× 100), mean and 95% CI over 10 seeds. No difference is significant after Holm correction.],
) <tab-ogm-synth>

The modulation does what it should: the text, which learns first, gets $k < 1$ ($k = 0.76$ with Adam, $alpha = 1$), and the other modalities keep $k approx 1$. But OM brings no gain (@tab-ogm-synth). The largest effect (independent, Adam, +1.1 Acc-2) appears with GE only, and is not significant after correction ($p_"Holm" = 0.06$); if real, it comes from the gradient noise acting as a regularizer, not from the rebalancing. Strong modulation hurts: with SGD and $alpha = 3$, the Acc-2 drops by 0.8 ($p_"Holm" < 0.05$).

Here the joint training does not under-train vision: a linear probe of the vision encoder gives the same Acc-2 trained jointly (67.7), alone (68.2), and *never trained* (67.5). With low-dimensional inputs, random features are already enough for a linear probe, so *a probe must always be compared with a random encoder*.

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
    [*Method*], [*$alpha$*], [Δ Acc-2], [Δ F1], [Δ Acc-2], [Δ F1],
    table.hline(stroke: 0.5pt),
    [Baseline], [], [78.00 ± 0.96], [78.02 ± 0.93], [77.06 ± 1.08], [77.07 ± 1.06],
    table.hline(stroke: 0.3pt),
    [OM], [0.1 †], ci("−0.13", "−0.45", "+0.13"), ci("−0.13", "−0.45", "+0.12"), ci("+0.13", "−0.39", "+0.71"), ci("+0.12", "−0.39", "+0.67"),
    [],   [0.3],   ci("−0.83", "−1.46", "−0.29"), ci("−0.84", "−1.46", "−0.30", strong: true), ci("+0.16", "−0.38", "+0.70"), ci("+0.15", "−0.38", "+0.68"),
    [],   [1],     ci("−0.32", "−1.31", "+0.66"), ci("−0.33", "−1.31", "+0.63"), ci("+0.74", "+0.12", "+1.49"), ci("+0.75", "+0.12", "+1.49"),
    [],   [3],     ci("−0.57", "−1.59", "+0.42"), ci("−0.58", "−1.56", "+0.37"), ci("+1.17", "+0.15", "+2.19"), ci("+1.19", "+0.19", "+2.20"),
    table.hline(stroke: 0.3pt),
    [OGM-GE], [0.1 †], ci("−0.38", "−0.89", "+0.13"), ci("−0.38", "−0.89", "+0.12"), ci("+0.76", "−0.07", "+1.71"), ci("+0.75", "−0.08", "+1.72"),
    [],       [0.3],   ci("−0.52", "−1.40", "+0.41"), ci("−0.51", "−1.37", "+0.39"), ci("+0.29", "−0.35", "+0.95"), ci("+0.30", "−0.34", "+0.96"),
    [],       [1],     ci("−0.20", "−0.86", "+0.52"), ci("−0.18", "−0.82", "+0.55"), ci("−0.04", "−0.61", "+0.54"), ci("−0.03", "−0.59", "+0.56"),
    [],       [3],     ci("−0.26", "−1.05", "+0.47"), ci("−0.25", "−1.00", "+0.45"), ci("+1.14", "+0.28", "+2.04"), ci("+1.17", "+0.32", "+2.06"),
    table.hline(),
  ),
  caption: [OGM − baseline on the CMU-MOSI test set (× 100). First row: absolute baseline. Mean and 95% CI over 10 paired seeds. †: $alpha$ selected on validation. Bold: significant after Holm correction ($p_"Holm" = 0.03$).],
) <tab-ogm-mosi>

*With Adam, OGM-GE never helps*: every difference is negative, and OM with $alpha = 0.3$ significantly lowers the F1 (−0.84, $p_"Holm" = 0.03$).

*With SGD, no gain is significant* (all $p_"Holm" >= 0.39$). The largest differences (+1.2 Acc-2 at $alpha = 3$) only bring SGD to the level of the Adam baseline (78.2 against 78.0). The selected $alpha = 0.1$ gives +0.1 (OM) and +0.8 (OGM-GE).

The MDE is 0.5 to 1.7 points of Acc-2 or F1: a real gain of about one point could be missed.

== Does OGM-GE rebalance the fusion?

*The text still dominates*: it provides 93–97% of the prediction in every condition (96% without OGM).

#figure(
  table(
    columns: 7,
    align: (left, left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Encoder*], [], [*Random*], [*Alone*], [*Joint*], [*+ OM*], [*+ OGM-GE*]),
    table.hline(stroke: 0.5pt),
    [Text],  [Acc-2], [68.4 ± 1.7], [77.0 ± 0.7], [77.8 ± 0.7], [77.3 ± 1.0], [77.7 ± 0.5],
    [],      [F1],    [68.4 ± 1.7], [76.9 ± 0.7], [77.8 ± 0.7], [77.3 ± 1.0], [77.7 ± 0.5],
    table.hline(stroke: 0.3pt),
    [Audio], [Acc-2], [53.6 ± 0.7], [54.0 ± 0.7], [54.5 ± 0.8], [54.6 ± 0.6], [54.8 ± 0.7],
    [],      [F1],    [50.0 ± 1.0], [51.1 ± 1.1], [52.2 ± 1.2], [52.4 ± 0.9], [52.6 ± 1.1],
    table.hline(stroke: 0.3pt),
    [Video], [Acc-2], [49.9 ± 1.6], [52.7 ± 1.5], [51.0 ± 1.1], [51.7 ± 0.5], [50.9 ± 1.1],
    [],      [F1],    [43.3 ± 3.9], [*50.6 ± 2.6*], [44.6 ± 2.3], [46.2 ± 1.9], [45.1 ± 2.3],
    table.hline(),
  ),
  caption: [Linear probe of each encoder on the CMU-MOSI test set (× 100, 10 seeds, Adam, $alpha = 0.1$). Random: initial weights, never trained. Always predicting the majority class gives an Acc-2 of 55.25.],
) <tab-ogm-probe10>

*Audio and video stay below the majority class.* No audio or video probe, trained or not, reaches the Acc-2 of the majority class (55.25). Only the text encoder learns something useful (Acc-2 68.4 → 77.0).

*The video encoder is under-trained inside the fusion, but at chance level.* Its F1 is 6.1 points lower trained jointly than alone ([−7.9, −4.4], 10 paired seeds), and no better than a random encoder (@tab-ogm-probe10). This confirms with 10 seeds the drop of @tab-ogm-probe. OM recovers part of it (+1.3 to +2.9 F1 with Adam), but the video stays below the majority class and its share in the fusion does not grow, so the fusion does not benefit.

*How much can audio and video add?* A linear regression on the predictions of the three uni-modal models, fitted on validation, improves over the text alone by only 0.7 Acc-2 [0.3, 1.0] and 0.7 F1 [0.3, 1.0]. It reaches 77.6 Acc-2, below our fusion without OGM (78.0). The room for improvement is very small.

== Why OGM-GE does not help here

OGM-GE rests on one assumption: the weak modalities _contain_ useful information, but learn it too slowly because the dominant one reduces the loss first. Our measures suggest that this assumption does not hold on CMU-MOSI with our features. We see four reasons, from the most to the least likely.

+ *Audio and video carry almost no polarity information.* No audio or video probe reaches the majority class, even trained alone, and an optimal combination of the uni-modal models gains only 0.7 point over the text. Speeding up the learning of a modality cannot help if that modality has nothing to learn. On the datasets of the paper (CREMA-D, Kinetics-Sounds), each modality alone is far above chance.
+ *Our encoders lose the temporal information.* We average the audio (COVAREP) and video (Facet) features over the whole clip. The cues of sentiment are often short (a smile, a change of pitch), and the mean erases them.
+ *The text is very strong and the dataset is small.* The text features alone give 77% Acc-2; with about 1,300 training clips, the weak encoders overfit before they learn anything general. The room left for the other modalities is small.
+ *Our training differs from the paper's.* The paper trains large encoders from raw signals for many epochs with SGD, where the imbalance has time to build up. We train small encoders on pre-extracted features for at most 20 epochs, mostly with Adam, which cancels most of the modulation (@sec-ogm-adam).

The adaptation to regression is not a likely cause: on synthetic data, the modulation slows down the dominant modality as intended.

== Conclusion

- On CMU-MOSI, OGM-GE brings *no significant gain* to the concatenation fusion, with Adam or SGD, for any $alpha$; with Adam it can even hurt.
- The imbalance targeted by the paper exists (the video encoder is under-trained inside the fusion), but it concerns a modality at chance level: correcting it cannot improve the fusion.
- *With Adam, the modulation is mostly cancelled*; OGM-GE with Adam mostly tests the GE noise.
- *Limits*: one dataset, small encoders on pre-extracted features, effects below the MDE (0.5 to 1.7 points) cannot be excluded.

== Perspectives

OGM-GE acts on the _speed_ at which each encoder learns. Since our weak encoders seem to lack information rather than speed, two directions look more promising.

- *Make each encoder learn on its own.* Rather than slowing down the text, one can give every encoder its own objective: an auxiliary uni-modal loss on each encoder, a uni-modal pretraining of each encoder before the fusion @du2023unimodal, or a modality dropout that hides the text during training and forces the fusion to use the other modalities. These methods target directly the under-training of the video encoder that we measured.
- *Give the weak modalities more information.* Sequence encoders (recurrent or attention-based) instead of a mean over time, or richer pre-trained features, should first bring the audio and video probes above the majority class. Only then can a rebalancing method like OGM-GE be expected to help. Running our implementation on a dataset where every modality is informative, such as CREMA-D, would also confirm that the negative result comes from the data and not from the method.
