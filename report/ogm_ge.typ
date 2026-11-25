= Applying OGM-GE to the concatenation fusion

== The method

OGM-GE @peng2022ogm changes the training of the encoders, not the architecture. At each training step, it measures how well each modality alone predicts the label. The encoder of the modality that predicts best receives a smaller gradient, so that the other encoders have more room to learn. It has two parts.

*On-the-fly gradient modulation (OGM).* The paper uses the part of the output that comes from one modality as its uni-modal prediction. With our linear head (@eq-concat), this is $c_m = W_m phi_m + b \/ 3$, as in the previous section. For a batch $B$, each modality gets a score $s_m$. In the paper, $s_m$ is the sum of the softmax probabilities given to the true class. For a regression, we replace this probability by the Gaussian likelihood of the true label:
$ s_m = sum_(i in B) exp(-(c_(m,i) - y_i)^2 / (2 sigma^2)), $
with $sigma = 1$ on CMU-MOSI. Like the probability of the true class, this value is between 0 and 1 for each sample, and close to 1 when the modality predicts well. The paper has two modalities and compares the score of one with the score of the other. We have three, so we compare the score of each modality with the mean score of the two others:
$ rho_m = s_m / (1/2 sum_(j != m) s_j). $
For two modalities, this is exactly the ratio of the paper. A modality with $rho_m > 1$ predicts better than the others. Its gradient is multiplied by a coefficient $k_m < 1$:
$ k_m = cases(1 - tanh(alpha rho_m) & "if" rho_m > 1, 1 & "otherwise,") $
where $alpha >= 0$ sets the strength of the modulation. This is applied to all the parameters of the encoder of modality $m$. The linear head is never modulated.

*Generalization enhancement (GE).* The paper argues that the noise of stochastic gradient descent helps generalization, and that multiplying the gradient by $k_m$ also reduces this noise. GE therefore adds a Gaussian noise to the modulated gradient $g_m$ of each encoder:
$ g_m <- k_m g_m + h_m, quad h_m tilde cal(N)(0, sigma_g^2 I), $
where $sigma_g$ is the standard deviation of the entries of the original gradient tensor. This follows the official code, which uses $sigma_g$ as a simple approximation of the covariance of the gradient noise. The noise is added to all encoders, including the ones that are not slowed down.

In our code, OGM-GE is an option of the fusion model (`--ogm none|ogm|ogm-ge`, with `--ogm-alpha`). The modulation is applied after the backward pass and before the optimizer step. With `--ogm none`, training is exactly the same as before.

== Setup

We keep the model of the previous section: the same encoders, seeds and early stopping on the validation MAE. We compare two optimizers:
- *Adam* with the previous protocol (learning rate $10^(-3)$, at most 20 epochs, patience 5). Without modulation, this is exactly the fusion of @tab-ogm-models.
- *SGD* as in the paper: momentum 0.9, weight decay $10^(-4)$ and learning rate $10^(-3)$. SGD learns more slowly, so we allow at most 100 epochs, with a patience of 10.

As in the paper, $alpha$ is chosen in $[0, 1]$ on the validation set. We try $alpha in {0.1, 0.3, 0.5, 0.7, 0.9}$ with OGM-GE and keep the value with the lowest mean validation MAE over the three seeds: $alpha = 0.5$ with Adam and $alpha = 0.9$ with SGD. To separate the two parts of the method, we also train the fusion with OGM alone (without noise), with the same $alpha$.

#figure(
  table(
    columns: 6,
    align: (left, left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Optimizer*], [*Modulation*], [*Acc-2*], [*F1*], [*Corr*], [*MAE*]),
    table.hline(stroke: 0.5pt),
    [Adam], [none],   [77.45 ± 0.72], [77.48 ± 0.74], [66.29 ± 0.27], [95.47 ± 1.20],
    [Adam], [OGM],    [77.55 ± 0.77], [77.56 ± 0.78], [67.30 ± 0.92], [*94.81 ± 4.90*],
    [Adam], [OGM-GE], [*78.67 ± 0.59*], [*78.71 ± 0.58*], [*68.02 ± 1.87*], [96.19 ± 2.36],
    table.hline(stroke: 0.3pt),
    [SGD], [none],    [77.07 ± 1.02], [77.10 ± 1.00], [67.26 ± 1.21], [*94.69 ± 2.44*],
    [SGD], [OGM],     [*78.52 ± 1.21*], [78.54 ± 1.18], [*68.16 ± 0.42*], [94.91 ± 1.27],
    [SGD], [OGM-GE],  [*78.52 ± 0.17*], [*78.56 ± 0.18*], [67.98 ± 0.83], [95.90 ± 1.35],
    table.hline(),
  ),
  caption: [Concatenation fusion on the CMU-MOSI test set, with and without gradient modulation (Acc-2 _Has0_, weighted F1, Corr × 100, MAE × 100). $alpha = 0.5$ with Adam and $0.9$ with SGD. Mean ± standard deviation over 3 seeds. Best value per optimizer in bold.],
) <tab-ogm-ge>

To see the effect on each modality, @tab-ogm-ge-enc gives the mean coefficient $k_T$ of the text during training, the share of the text in the fused prediction (defined in the previous section), and the correlation obtained by each encoder tested alone with a new linear layer, as in @tab-ogm-probe. Audio and video always get a coefficient close to 1 (between 0.94 and 1.00).

#figure(
  table(
    columns: 7,
    align: (left, left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Optimizer*], [*Modulation*], [*$k_T$*], [*Text share*], [*Text*], [*Audio*], [*Video*]),
    table.hline(stroke: 0.5pt),
    [Adam], [none],   [1],    [95.3 ± 2.7], [66.45 ± 0.23], [22.73 ± 0.87], [12.67 ± 1.92],
    [Adam], [OGM],    [0.34], [96.6 ± 0.2], [67.04 ± 0.66], [23.89 ± 0.44], [12.05 ± 2.36],
    [Adam], [OGM-GE], [0.36], [95.1 ± 0.3], [67.73 ± 1.97], [25.34 ± 0.57], [13.30 ± 0.69],
    table.hline(stroke: 0.3pt),
    [SGD], [none],    [1],    [96.9 ± 1.2], [66.62 ± 1.16], [24.68 ± 0.32], [10.80 ± 3.04],
    [SGD], [OGM],     [0.17], [95.7 ± 0.6], [67.45 ± 0.50], [23.99 ± 0.56], [10.17 ± 1.91],
    [SGD], [OGM-GE],  [0.17], [95.0 ± 0.3], [67.18 ± 0.85], [24.51 ± 0.20], [13.36 ± 2.59],
    table.hline(stroke: 0.3pt),
    table.cell(colspan: 4)[_Encoder trained alone (Adam, @tab-ogm-probe)_], [67.70 ± 0.85], [23.46 ± 1.13], [15.93 ± 2.87],
    table.hline(),
  ),
  caption: [Effect of the modulation on each modality (CMU-MOSI test set). $k_T$: mean coefficient of the text encoder during training. Text share: part of the variance of the fused prediction that comes from the text (%). Last three columns: correlation (× 100) of each encoder of the fusion, frozen, with a new linear layer. Mean ± standard deviation over 3 seeds.],
) <tab-ogm-ge-enc>

== Interpretation

*The modulation targets the text, as expected.* During training, the text gets a ratio $rho_T$ of about 1.5 to 1.6: its partial prediction is clearly better than those of audio and video. Its gradient is therefore reduced, on average to 34--36% with Adam and to 17% with SGD (the selected $alpha$ is larger with SGD). Audio and video are almost never slowed down.

*The fusion gains a little in classification, but not in regression.* With both optimizers, OGM-GE improves the Acc-2 (+1.2 points with Adam, +1.5 with SGD) and the correlation (+1.7 and +0.7). Its Acc-2 is also more stable with SGD (standard deviation of 0.17). However, the MAE does not improve: it is even slightly worse (96.19 against 95.47 with Adam). On the validation set, which we used to choose $alpha$, the gain is also very small: with Adam, no value of $alpha$ gives a lower validation MAE than the fusion without modulation (89.85 at best against 89.56). The differences are about one or two standard deviations, with only three seeds. The gain is therefore small, and we cannot be sure that it is real.

*With Adam, OGM alone does almost nothing.* Without the noise, OGM gives the same Acc-2 as the baseline with Adam (77.55 against 77.45), but a gain of 1.45 points with SGD. A likely reason is that Adam divides each gradient by a running estimate of its own size. Multiplying the gradient of the text encoder by a coefficient that changes slowly is then largely cancelled, and the size of the update barely changes. With SGD, the update is directly proportional to the gradient, so OGM really slows down the text encoder. This agrees with the paper, where OGM-GE also helps less with Adam than with SGD (their Table 6). With Adam, the gain of OGM-GE comes mainly from the added noise.

*The text still dominates.* In all cases, about 95% of the fused prediction still comes from the text (@tab-ogm-ge-enc). OGM-GE lowers this share by less than 2 points. Slowing down the text encoder does not make the head use audio and video much more.

*The encoders are only slightly better trained.* The video encoder of the fusion improves with OGM-GE: its correlation goes from 12.67 to 13.30 with Adam and from 10.80 to 13.36 with SGD. This goes in the direction of the paper. But it stays below the video encoder trained alone (15.93), and the standard deviations are large (up to 2.6). OGM alone does not improve the video encoder: the gain appears only with the noise. For the audio, OGM-GE helps with Adam (22.73 to 25.34, even above the encoder trained alone) but not with SGD (24.68 to 24.51). The text encoder is not damaged by being slowed down: its correlation is even slightly higher (about 67 to 68 instead of 66.5).

*Why the effect is limited.* OGM-GE only changes how fast each encoder learns. It cannot add information. On CMU-MOSI, the audio and video features carry little information: trained alone, they do not beat the majority class (@tab-ogm-models). Even a well-trained video encoder therefore brings little to the fusion. In the paper, on the contrary, both modalities are informative: on CREMA-D (6 classes), the audio alone reaches 52.5% accuracy and the images alone 41.9%, far above chance (about 17%). There is then more to gain by balancing them, and OGM-GE improves the concatenation fusion from 51.7% to 61.9%.

*Limits.* We use three seeds only, and most differences are within one or two standard deviations. Our score $s_m$ is an adaptation of the paper's score to regression, and other choices are possible. With SGD, the selected $alpha$ (0.9) is at the edge of the interval $[0, 1]$ given in the paper, so a stronger modulation might do better on validation. Finally, our setting is different from the paper's (regression, pre-extracted features, small MLP encoders).
