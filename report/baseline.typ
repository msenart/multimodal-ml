= Baselines

== Models

We trained three reference models with their default hyperparameters in MMSA @mao2022msena. They represent three ways of fusing the modalities.

- *LF-DNN* (Late Fusion DNN) @williams2018lfdnn. Each modality is encoded separately. The three representations are concatenated at the end, then passed to a small network that predicts the score. This is a simple late fusion.
- *TFN* (Tensor Fusion Network) @zadeh2017tfn. Each modality is first summarized by a vector. A constant component equal to 1 is appended to each vector, and the outer product of the three vectors is computed. The resulting tensor contains the unimodal, bimodal and trimodal terms at the same time.
- *MulT* (Multimodal Transformer) @tsai2019mult. It uses cross-modal attention: each modality "queries" the two others at every time step. MulT therefore does not need the sequences to be aligned beforehand, which suits our data.

Each model is trained with three random seeds (1111, 1112 and 1113). We report the mean and the standard deviation over these three runs.

== Modality ablation

To measure the contribution of each modality, we also train the same models on a single modality: text only (T), audio only (A) or video only (V). The removed modalities are replaced with zeros (and, for the text, the token ids are replaced with the `[PAD]` token). The architecture does not change: only the available information changes. The full version is denoted TAV.

== Large language models

Finally, we compare these models with two large language models (LLMs), Qwen3.8-27B and DeepSeek-Chat. They are used _zero-shot_: they are not trained on our data and only see the transcript. They are asked to rate the sentiment of each clip on the scale of the dataset ($[-3, 3]$ for CMU-MOSI, $[-1, 1]$ for CH-SIMS). Clips are sent in batches of 10, with a temperature of 0. Answers are clipped to the scale interval if needed. The metrics are then computed as for the other models. These are single runs, without standard deviation.

= Results

== CMU-MOSI

@tab-mosi gives the results on the CMU-MOSI test set.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Modalities*], [*Acc-2*], [*F1*], [*Corr*]),
    table.hline(stroke: 0.5pt),
    [LF-DNN], [A],   [44.75 ± 0.00], [27.67 ± 0.00], [10.18 ± 7.73],
    [TFN],    [A],   [49.66 ± 3.61], [40.97 ± 10.08], [15.02 ± 0.96],
    [MulT],   [A],   [53.89 ± 1.79], [50.60 ± 3.25], [23.35 ± 0.94],
    table.hline(stroke: 0.3pt),
    [LF-DNN], [V],   [49.27 ± 6.39], [37.92 ± 14.50], [12.85 ± 3.44],
    [TFN],    [V],   [51.51 ± 2.21], [46.20 ± 5.51], [12.00 ± 2.00],
    [MulT],   [V],   [57.68 ± 2.48], [57.22 ± 2.41], [17.78 ± 0.87],
    table.hline(stroke: 0.3pt),
    [LF-DNN], [T],   [76.72 ± 1.04], [76.68 ± 0.99], [64.90 ± 0.74],
    [TFN],    [T],   [76.38 ± 1.57], [76.41 ± 1.53], [66.25 ± 1.37],
    [MulT],   [T],   [78.52 ± 0.38], [78.50 ± 0.40], [66.49 ± 1.06],
    table.hline(stroke: 0.3pt),
    [LF-DNN], [TAV], [76.58 ± 1.01], [76.52 ± 0.89], [65.11 ± 0.73],
    [TFN],    [TAV], [76.68 ± 0.41], [76.69 ± 0.39], [65.64 ± 0.75],
    [MulT],   [TAV], [79.79 ± 0.25], [79.70 ± 0.30], [70.04 ± 0.79],
    table.hline(stroke: 0.3pt),
    [Qwen3.8-27B (zero-shot)],   [T], [*88.63*], [*88.66*], [*89.30*],
    [DeepSeek-Chat (zero-shot)], [T], [88.05], [88.08], [85.83],
    table.hline(),
  ),
  caption: [Results on the CMU-MOSI test set (Acc-2 _Has0_, weighted F1, Corr × 100). Mean ± standard deviation over 3 seeds. Always predicting the majority class gives an Acc-2 of 55.25.],
) <tab-mosi>

*Text is the most useful modality.* With text only, the three models already reach 76 to 79% Acc-2. With audio only or video only, they stay close to chance and below the majority class (55.25%). Only MulT on video goes slightly above it (57.68%). The case of LF-DNN on audio is telling: its standard deviation is zero and its Acc-2 (44.75%) is exactly the proportion of positive samples in the test set. The model therefore always predicts "positive". It has learned the majority class of the training set (57% positive), which is the minority class in the test set. Its very low F1 (27.67) also shows this.

*Fusion only helps MulT.* For LF-DNN and TFN, the TAV and T results are almost identical (the differences are smaller than the standard deviations). These two models therefore do not make use of audio and video. MulT, on the other hand, gains 1.27 points of Acc-2 and 3.55 points of correlation when going from T to TAV. It is also the best model in every configuration, and the most stable one (standard deviation of 0.25 in TAV). Its cross-modal attention seems to make better use of the non-text, unaligned modalities.

*Zero-shot LLMs do much better.* Without any training and with text only, Qwen3.8-27B and DeepSeek-Chat outperform the best multimodal model by 8.8 and 8.3 points of Acc-2 (88.63 and 88.05 vs. 79.79). Qwen3.8-27B also gets the best correlation (89.30) and the lowest MAE (62.56 vs. 89.78 for MulT TAV). This result must however be qualified. These LLMs were pre-trained on very large amounts of text, and we cannot rule out that the CMU-MOSI transcripts were part of it. Each model was also run only once.

== CH-SIMS

@tab-sims gives the results of the LLMs on the CH-SIMS test set. The MMSA baselines have not been run on this dataset yet.

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Modalities*], [*Acc-2*], [*F1*], [*Corr*]),
    table.hline(stroke: 0.5pt),
    [Qwen3.8-27B (zero-shot)],   [T], [72.43], [73.37], [60.72],
    [DeepSeek-Chat (zero-shot)], [T], [*75.27*], [*76.02*], [*64.01*],
    table.hline(),
  ),
  caption: [Results on the CH-SIMS test set (Acc-2 with $0$ counted as negative, weighted F1, Corr × 100). Always predicting the majority class gives an Acc-2 of 69.37.],
) <tab-sims>

The results are much lower than on CMU-MOSI. Here too, they must be compared with the majority class: 69.37% of the test samples have a score $<= 0$. Qwen3.8-27B beats it by only 3.1 points of Acc-2, and DeepSeek-Chat by 5.9 points. The gap with the majority class is clearer on the F1 (73.37 and 76.02, vs. 56.8). Two reasons can explain this drop. First, in CH-SIMS, the text alone is not always enough to recover the multimodal label. The dataset provides a label annotated from the text only: its sign matches the sign of the multimodal label for only 83.6% of the test samples. A model that only sees the transcript is therefore limited, even if it understands the text perfectly. Second, many samples are neutral (15% of the test samples have a score of exactly zero). They are counted as negative, so a model that gives them a slightly positive score makes an error.
