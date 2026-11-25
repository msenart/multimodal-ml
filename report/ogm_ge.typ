= Balancing the modalities with OGM-GE

The baselines show a strong imbalance between the modalities on CMU-MOSI: text alone gives almost the same results as the three modalities together, and audio and video alone stay close to chance. A common explanation in the literature is that the dominant modality is learned faster, so the joint loss decreases before the encoders of the other modalities are well trained. OGM-GE (On-the-fly Gradient Modulation with Generalization Enhancement) @peng2022ogm was designed for this case. It slows down the learning of the dominant modality during training, so that the weaker modalities have more time to learn. In this section we test whether this method helps on CMU-MOSI.

== Method

*Model.* We use a simple late-fusion model, as in the paper. Each modality $u in {T, A, V}$ is averaged over time, then encoded by one linear layer with a ReLU: $h^u = "ReLU"(W_e^u macron(x)^u)$, with 128 dimensions. A single linear layer on the concatenation $[h^T; h^A; h^V]$ predicts the score. Its weight splits into one block $W^u$ per modality, so the prediction is a sum of one term per modality:
$ hat(y) = sum_u hat(y)^u, quad hat(y)^u = W^u h^u + b / 3. $
The loss is the mean squared error of $hat(y)$.

*Gradient modulation (OGM).* The paper is written for classification with two modalities. We adapt it to regression with three modalities. For each mini-batch $B$, the contribution of modality $u$ is measured by
$ S^u = sum_(i in B) exp(-|hat(y)_i^u - y_i|), $
which is larger when the term $hat(y)^u$ alone is closer to the label. This replaces the softmax probability of the true class used in the paper. The discrepancy ratio compares $u$ with the mean of the two other modalities:
$ rho^u = S^u / (1/2 sum_(v != u) S^v). $
After the backward pass, the gradient of the encoder of $u$ is multiplied by
$ k^u = cases(1 - tanh(alpha rho^u) & "if" rho^u > 1, 1 & "otherwise,") $
where $alpha >= 0$ sets the strength of the modulation. A modality that contributes more than the others ($rho^u > 1$) is therefore slowed down. The classifier is not modulated.

*Generalization enhancement (GE).* Reducing a gradient also reduces its noise, and the paper shows that this hurts generalization. GE adds back Gaussian noise: $tilde(g)^u = k^u g^u + epsilon$, with $epsilon tilde.op cal(N)(0, hat(sigma)^2)$. We estimate the variance $hat(sigma)^2$ of each parameter's gradient with an exponential moving average over mini-batches (decay 0.9), instead of the per-sample covariance of the paper.

*Optimizer.* As in the paper, we use SGD with momentum 0.9 and weight decay $10^(-4)$. Adam is not suited to this method: it divides each gradient by a running estimate of its magnitude, so multiplying a gradient by a constant $k^u$ has almost no effect.

== Experimental protocol

The method changes several things at once compared with a plain model: the gradient modulation, the noise and the optimizer. To know which part causes a change, we train five variants. They share exactly the same architecture and training loop and differ only in the following switches:
- *Linear (Adam)*: no modulation, no noise, Adam. This is a standard late-fusion baseline;
- *Linear (SGD)*: no modulation, no noise, SGD;
- *GE only*: noise, no modulation, SGD;
- *OGM only*: modulation, no noise, SGD;
- *OGM-GE*: modulation and noise, SGD (the full method).

All variants are trained for at most 100 epochs with batches of 32, with early stopping on the validation MAE (patience 15). The test set is evaluated once, on the checkpoint with the best validation MAE. Each configuration is trained with 10 random seeds (1111 to 1120). For a given seed, all variants start from the same initial weights and see the first epoch in the same order, so the comparisons are paired.

All hyperparameters are chosen on the validation set only. First, the learning rate of each variant is chosen among ${3 dot 10^(-5), 10^(-4), 3 dot 10^(-4), 10^(-3), 3 dot 10^(-3)}$ for Adam and ${3 dot 10^(-4), 10^(-3), 3 dot 10^(-3), 10^(-2), 3 dot 10^(-2), 10^(-1)}$ for SGD (with $alpha = 0.5$ for the two OGM variants). We keep the one with the lowest mean validation MAE over the 10 seeds. Second, the two OGM variants are retrained at their chosen learning rate with $alpha in {0.1, 0.3, 0.5, 0.8, 1.0}$, and we keep the $alpha$ with the lowest mean validation MAE. In total, 450 models were trained.

With SGD, a learning rate of $10^(-1)$ makes all four SGD variants diverge for all seeds, and $3 dot 10^(-2)$ is unstable. The chosen learning rates are $10^(-4)$ for Adam and $3 dot 10^(-4)$ for all SGD variants. With SGD, $3 dot 10^(-4)$ and $10^(-3)$ give the same validation MAE up to the noise (93.11 vs. 93.25 for Linear (SGD)). The chosen $alpha$ is 0.1 for both OGM variants.

== Results

=== Controlled comparison

@tab-ogm-main gives the test results of the five variants.

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Variant*], [*lr*], [*$alpha$*], [*Acc-2*], [*F1*], [*MAE*], [*Corr*]),
    table.hline(stroke: 0.5pt),
    [Linear (Adam)], [$10^(-4)$],      [–],   [77.94 ± 0.61], [78.00 ± 0.61], [*96.99 ± 0.35*], [67.62 ± 0.37],
    [Linear (SGD)],  [$3 dot 10^(-4)$], [–],   [*78.53 ± 0.52*], [*78.58 ± 0.52*], [97.63 ± 1.05], [67.55 ± 0.42],
    [GE only],       [$3 dot 10^(-4)$], [–],   [78.43 ± 0.77], [78.48 ± 0.77], [97.56 ± 0.80], [*67.67 ± 0.32*],
    [OGM only],      [$3 dot 10^(-4)$], [0.1], [78.27 ± 0.77], [78.32 ± 0.77], [97.81 ± 0.99], [67.48 ± 0.38],
    [OGM-GE],        [$3 dot 10^(-4)$], [0.1], [78.41 ± 0.84], [78.47 ± 0.83], [97.66 ± 0.94], [67.58 ± 0.46],
    table.hline(),
  ),
  caption: [Test results on CMU-MOSI of the five variants (Acc-2 _Has0_, weighted F1, MAE × 100, Corr × 100). Mean ± standard deviation over 10 seeds. Learning rate and $alpha$ chosen on the validation set.],
) <tab-ogm-main>

*OGM-GE brings no measurable gain.* The five variants are within 0.6 points of Acc-2 and 0.2 points of correlation of each other, which is smaller than the standard deviation over seeds. To check this, we use paired tests over the 10 seeds (@tab-ogm-tests). Compared with the same model trained with SGD and no modulation, OGM-GE changes the Acc-2 by $-0.12$ points ($p = 0.54$), the MAE by $+0.03$ ($p = 0.89$) and the correlation by $+0.03$ ($p = 0.68$). The modulation alone (OGM-GE vs. GE only) and the noise alone (OGM-GE vs. OGM only) have no significant effect either. The only difference with $p < 0.05$ is that OGM-GE has a slightly higher MAE than Linear (Adam) ($+0.67$, $p = 0.035$). We ran 12 tests here, so a Bonferroni correction gives a threshold of $0.05 \/ 12 approx 0.004$, and this difference is not significant either. It would in any case go against OGM-GE.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*OGM-GE vs.*], [*$Delta$ Acc-2*], [*$Delta$ MAE*], [*$Delta$ Corr*]),
    table.hline(stroke: 0.5pt),
    [Linear (Adam)], [+0.47 ($p = 0.20$)], [+0.67 ($p = 0.035$)], [$-$0.03 ($p = 0.80$)],
    [Linear (SGD)],  [$-$0.12 ($p = 0.54$)], [+0.03 ($p = 0.89$)], [+0.03 ($p = 0.68$)],
    [GE only],       [$-$0.01 ($p = 0.88$)], [+0.10 ($p = 0.65$)], [$-$0.09 ($p = 0.34$)],
    [OGM only],      [+0.15 ($p = 0.33$)], [$-$0.15 ($p = 0.57$)], [+0.11 ($p = 0.21$)],
    table.hline(),
  ),
  caption: [Mean paired difference (OGM-GE minus the other variant) over the 10 seeds, with the $p$-value of a paired $t$-test. A Wilcoxon signed-rank test gives the same conclusions. A lower MAE is better.],
) <tab-ogm-tests>

=== Comparison with the baselines

@tab-ogm-ref puts our models next to the MMSA baselines of @tab-mosi. All results use the same test set and the same metrics. Our simple linear late-fusion models are better than LF-DNN and TFN in TAV (+1.3 to +2.0 points of Acc-2 and +1.9 to +2.5 points of correlation), but below MulT ($-1.3$ to $-1.9$ points of Acc-2 and about $-2.5$ points of correlation). OGM-GE does not change this ranking. The comparison has two limits: the MMSA models were trained with 3 seeds and their default hyperparameters, while our models were tuned on the validation set and trained with 10 seeds.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Acc-2*], [*F1*], [*MAE*], [*Corr*]),
    table.hline(stroke: 0.5pt),
    [LF-DNN (MMSA)], [76.58 ± 1.01], [76.52 ± 0.89], [98.35 ± 2.80], [65.11 ± 0.73],
    [TFN (MMSA)],    [76.68 ± 0.41], [76.69 ± 0.39], [96.47 ± 1.47], [65.64 ± 0.75],
    [MulT (MMSA)],   [*79.79 ± 0.25*], [*79.70 ± 0.30*], [*89.78 ± 0.87*], [*70.04 ± 0.79*],
    table.hline(stroke: 0.3pt),
    [Linear (Adam)], [77.94 ± 0.61], [78.00 ± 0.61], [96.99 ± 0.35], [67.62 ± 0.37],
    [Linear (SGD)],  [78.53 ± 0.52], [78.58 ± 0.52], [97.63 ± 1.05], [67.55 ± 0.42],
    [OGM-GE],        [78.41 ± 0.84], [78.47 ± 0.83], [97.66 ± 0.94], [67.58 ± 0.46],
    table.hline(),
  ),
  caption: [Comparison with the MMSA baselines on the CMU-MOSI test set, all with the three modalities (TAV). MMSA: 3 seeds; ours: 10 seeds.],
) <tab-ogm-ref>

=== Effect of $alpha$

@tab-ogm-alpha shows the effect of the modulation strength. A larger $alpha$ slows down the text encoder more: at the best epoch, the mean coefficient of the text goes from $k^T = 0.84$ ($alpha = 0.1$) to $0.10$ ($alpha = 1$). The validation MAE increases with $alpha$, from 93.2 to 95.5, and so does the test MAE. The test correlation decreases (from 67.6 to 66.2 for OGM-GE). The Acc-2 changes by less than one standard deviation. The stronger the modulation, the worse the model: slowing down the text does not help the model, it only slows down the learning of its most useful modality.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Variant*], [*$alpha$*], [*$k^T$*], [*Val. MAE*], [*Test Acc-2*], [*Test Corr*]),
    table.hline(stroke: 0.5pt),
    [OGM-GE], [0.1], [0.84], [*93.18 ± 2.07*], [*78.41 ± 0.84*], [*67.58 ± 0.46*],
    [OGM-GE], [0.3], [0.55], [93.49 ± 2.19], [78.29 ± 0.70], [67.39 ± 0.57],
    [OGM-GE], [0.5], [0.33], [93.91 ± 2.18], [78.27 ± 0.95], [67.14 ± 0.56],
    [OGM-GE], [0.8], [0.16], [94.73 ± 2.19], [78.06 ± 0.83], [66.56 ± 0.67],
    [OGM-GE], [1.0], [0.10], [95.52 ± 2.18], [77.99 ± 0.89], [66.17 ± 0.63],
    table.hline(stroke: 0.3pt),
    [OGM only], [0.1], [0.84], [*93.18 ± 1.92*], [78.27 ± 0.77], [*67.48 ± 0.38*],
    [OGM only], [0.3], [0.55], [93.55 ± 2.00], [*78.59 ± 0.86*], [*67.48 ± 0.43*],
    [OGM only], [0.5], [0.34], [93.88 ± 2.13], [78.22 ± 0.63], [67.27 ± 0.39],
    [OGM only], [0.8], [0.15], [94.74 ± 2.35], [78.12 ± 0.94], [66.83 ± 0.38],
    [OGM only], [1.0], [0.09], [95.47 ± 2.47], [77.87 ± 0.92], [66.32 ± 0.80],
    table.hline(),
  ),
  caption: [Effect of $alpha$ (learning rate $3 dot 10^(-4)$, 10 seeds). $k^T$: mean modulation coefficient of the text encoder at the best epoch. MAE and Corr × 100.],
) <tab-ogm-alpha>

=== Diagnostics: does OGM-GE reduce the imbalance?

The method could fail in two ways: either it does not reduce the imbalance during training, or it reduces it but this does not help the predictions. We use three diagnostics to tell these cases apart.

*Training dynamics.* @fig-ogm-dynamics follows the discrepancy ratio $rho^u$ and the coefficient $k^u$ during training. Without modulation, $rho^T$ starts at 1 and grows steadily to about 1.75 after 40 epochs: the text term quickly becomes much closer to the label than the audio and video terms. With OGM-GE, the text encoder is slowed down ($k^T$ decreases to about 0.83), while $k^A$ and $k^V$ stay at 1 after the first epochs. However, the curves of $rho^u$ with and without modulation almost overlap. The modulation therefore does not reduce the imbalance: the audio and video terms do not become better predictors when the text is slowed down.

#figure(
  image("figures/ogm_ge_dynamics.svg", width: 100%),
  caption: [Training dynamics, mean over the 10 seeds (selected configurations; an epoch is shown while at least half of the runs are still training). (a) Discrepancy ratio $rho^u$ without modulation (Linear (SGD), solid) and with OGM-GE (dashed); the dotted line is $rho = 1$. (b) Modulation coefficient $k^u$ of OGM-GE.],
) <fig-ogm-dynamics>

*Masked modalities.* At test time, we replace some modalities with zeros, which is their training mean since the features are standardized. @tab-ogm-masked shows that all models rely almost only on the text. Removing audio and video costs at most 0.5 points of Acc-2 and 0.7 points of correlation, with or without OGM-GE. Linear (Adam) even gets the same Acc-2 with text only as with the three modalities. With audio and video only, all models are close to chance (about 53% Acc-2 and a correlation of 23–24). OGM-GE does not make the model use audio and video more.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Modalities kept*], [*Linear (SGD) Acc-2*], [*Linear (SGD) Corr*], [*OGM-GE Acc-2*], [*OGM-GE Corr*]),
    table.hline(stroke: 0.5pt),
    [TAV], [78.53 ± 0.52], [67.55 ± 0.42], [78.41 ± 0.84], [67.58 ± 0.46],
    [T],   [78.03 ± 0.41], [66.93 ± 0.39], [77.94 ± 0.57], [66.93 ± 0.34],
    [A],   [50.63 ± 2.11], [24.40 ± 0.75], [50.61 ± 1.89], [24.26 ± 0.87],
    [V],   [49.50 ± 1.18], [12.99 ± 2.34], [49.55 ± 1.42], [13.33 ± 2.37],
    [AV],  [52.99 ± 2.20], [23.38 ± 1.52], [52.99 ± 2.24], [23.57 ± 1.45],
    table.hline(),
  ),
  caption: [Test results when only some modalities are kept at inference (the others are set to zero). Mean ± standard deviation over 10 seeds.],
) <tab-ogm-masked>

*Linear probes.* The goal of OGM-GE is to train better encoders for the weak modalities. To measure this directly, we freeze the trained encoders and fit, for each modality, a ridge regression from $h^u$ to the label on the training set (the ridge penalty is chosen on the validation set). @tab-ogm-probes gives the test results. The audio and video encoders reach the same quality with and without modulation: a correlation of about 22 for audio and 12 for video. The small difference on audio (22.29 vs. 21.45) also appears with GE only (22.29), so it comes from the noise, not from the modulation. OGM only gives almost the same probe results as Linear (SGD) (audio: 21.45 in both cases).

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Encoder*], [*Linear (SGD) Acc-2*], [*Linear (SGD) Corr*], [*OGM-GE Acc-2*], [*OGM-GE Corr*]),
    table.hline(stroke: 0.5pt),
    [Text],   [78.03 ± 0.41], [66.96 ± 0.40], [77.97 ± 0.44], [66.97 ± 0.34],
    [Audio],  [54.10 ± 0.92], [21.45 ± 3.17], [54.20 ± 0.90], [22.29 ± 2.72],
    [Video],  [50.15 ± 1.62], [11.68 ± 4.38], [50.23 ± 1.72], [11.71 ± 4.38],
    table.hline(),
  ),
  caption: [Linear probes: ridge regression on the frozen output $h^u$ of each encoder, evaluated on the test set. Mean ± standard deviation over 10 seeds.],
) <tab-ogm-probes>

== Discussion

On CMU-MOSI, OGM-GE does not improve the results and does not reduce the imbalance between the modalities. Our diagnostics suggest why. OGM-GE assumes that the weak modalities are *under-trained*: they contain useful information, but the dominant modality takes over before their encoders learn it. Here, the audio and video encoders reach the same quality whether the text is slowed down or not (@tab-ogm-probes), and this quality is close to that of the MMSA models trained on audio or video alone (correlation 23.35 and 17.78 for MulT, @tab-mosi). The weakness of audio and video therefore seems to come from the *information* they carry, at least in the form we use, and not from the way they are trained. Slowing down the text cannot add information to the other modalities. It can only delay the learning of the text, which is what we observe when $alpha$ increases (@tab-ogm-alpha).

Two more facts support this view. First, MulT, which keeps the time sequences and uses cross-modal attention, is the only model that gains from audio and video (@tab-mosi). Second, our encoders average the audio and video features over time. This loses the temporal dynamics (changes in the voice or in the face), which are probably the most useful part of these signals for sentiment. A balancing method like OGM-GE should therefore be combined with richer unimodal encoders to have a chance to help.

Our study has some limits. We use only CMU-MOSI and one simple architecture. Our adaptation of the method to regression and to three modalities (contribution score, ratio against the mean of the other modalities, noise estimated with a moving average) is our own and may differ from what the authors would do. Finally, $alpha$ was chosen at a fixed learning rate.
