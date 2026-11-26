= Reproducing the modality imbalance of OGM-GE

== The problem studied in the paper

Peng et al. @peng2022ogm study multimodal models trained with a single objective for all modalities. They observe that, even when the multimodal model is better than each uni-modal model, the encoder of each modality inside it is _worse_ than the same encoder trained alone. To measure this, they freeze an encoder of the jointly trained model and train only a new classifier on its features. Their explanation is an _optimization imbalance_: the modality that is the most useful for the task (the _dominant_ modality) reduces the loss quickly, and the other modalities then receive too little gradient to be trained well. They propose OGM-GE, a method that slows down the training of the dominant modality. In this part, we do not implement OGM-GE: we only check whether the imbalance it targets also appears on CMU-MOSI.

The paper analyses the simplest fusion: the features of the encoders are concatenated and passed to one linear layer. We use the same fusion, adapted to our regression task. Let $phi_T$, $phi_A$ and $phi_V$ be the outputs of the text, audio and video encoders. The predicted score is
$ hat(y) = W [phi_T; phi_A; phi_V] + b = W_T phi_T + W_A phi_A + W_V phi_V + b, $ <eq-concat>
where $W = [W_T, W_A, W_V]$ is split into one block per modality. With the mean squared error $L = 1/N sum_i (hat(y)_i - y_i)^2$, the gradient that reaches the encoder of modality $m$ (with parameters $theta_m$) is
$ (partial L) / (partial theta_m) = 2/N sum_(i=1)^N (hat(y)_i - y_i) W_m (partial phi_m (x_i)) / (partial theta_m). $
All modalities share the same error $hat(y)_i - y_i$. If the text alone already makes this error small, the audio and video encoders receive a small gradient, even if they have not learned much yet. This is the regression version of the argument of @peng2022ogm, who use a softmax and the cross-entropy loss.

Our setting differs from the paper in several ways: the task is a regression and not a classification, the inputs are pre-extracted features and not raw images and sounds, the encoders are small MLPs and not ResNet-18 networks, and we train with Adam instead of SGD.

== Uni-modal encoders

The three encoders have the same structure. They first turn the sequence of a modality into a single vector, then apply a small neural network:
- *Text.* The 768-dimensional BERT embeddings are averaged over the 50 positions of the sequence.
- *Audio* and *video.* The 5-dimensional (audio) and 20-dimensional (video) vectors are averaged over the real frames of the clip. The padding frames are ignored.
- The resulting vector goes through 3 layers of 256 units (each layer is linear, followed by a ReLU and a dropout of 0.3). The output of these layers, a 256-dimensional vector, is the representation $phi_m$ of the modality.
- For a uni-modal model, a final linear layer maps $phi_m$ to the predicted score.

Before training, each feature is standardized with the mean and the standard deviation of the training set. Averaging over time loses the order of the frames. This is a simple choice, also made by TFN @zadeh2017tfn for the audio and the video.

All models are trained the same way: mean squared error, Adam with a learning rate of $10^(-3)$, batches of 16 clips, at most 20 epochs. Training stops when the validation MAE has not improved for 5 epochs, and we keep the model with the best validation MAE. As for the baselines, we use three seeds (1111, 1112 and 1113) and report the mean and the standard deviation.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Acc-2*], [*F1*], [*Corr*]),
    table.hline(stroke: 0.5pt),
    [Text alone (T)],    [*77.50 ± 0.59*], [*77.56 ± 0.58*], [*67.45 ± 1.35*],
    [Audio alone (A)],   [54.66 ± 2.59], [51.72 ± 5.15], [23.62 ± 0.95],
    [Video alone (V)],   [52.67 ± 1.13], [49.06 ± 2.78], [15.99 ± 3.04],
    table.hline(stroke: 0.3pt),
    [Concatenation fusion (TAV)], [77.45 ± 0.72], [77.48 ± 0.74], [66.29 ± 0.27],
    table.hline(),
  ),
  caption: [Uni-modal encoders and concatenation fusion on the CMU-MOSI test set (Acc-2 _Has0_, weighted F1, Corr × 100). Mean ± standard deviation over 3 seeds. Always predicting the majority class gives an Acc-2 of 55.25.],
) <tab-ogm-models>

@tab-ogm-models (top) gives the results of the encoders trained alone. They are close to those of the MMSA models in @tab-mosi. The text is by far the most useful modality. The audio and the video carry a weak signal (correlations of 23.6 and 16.0), but their Acc-2 stays below the majority class.

== Concatenation fusion

The fusion model contains the three encoders described above. Their representations are concatenated into a vector of $3 times 256$ values, and one linear layer predicts the score (@eq-concat). This is the concatenation model of @peng2022ogm. The whole model is trained from scratch, with the same procedure and the same seeds as the uni-modal models.

The fusion reaches the same level as the text alone (@tab-ogm-models, bottom): 77.45% of Acc-2 against 77.50%, and a slightly lower correlation (66.29 against 67.45). Adding the audio and the video therefore brings almost nothing.

Since the head is linear, the prediction is a sum of three parts, one per modality: $hat(y) = c_T + c_A + c_V$, with $c_m = W_m phi_m + b \/ 3$. This lets us see how much each modality is used inside the fusion. We measure the _share_ of each modality: the variance of $c_m$ over the test set, divided by the sum of the three variances. The text gets $95.3 ± 2.7%$ of the share, the video $3.8 ± 2.7%$ and the audio $1.0 ± 0.3%$. The prediction of the fusion therefore depends almost only on the text.

== Encoders trained jointly, tested alone

We now follow the protocol of @peng2022ogm to evaluate each encoder on its own. We freeze the encoder and train only a new linear layer on its representation $phi_m$ (Adam, at most 200 epochs, early stopping on the validation MAE). We do this for two versions of each encoder: the one trained alone, and the one trained inside the fusion. The first version is a control: its result should be close to the uni-modal model, and it is (67.69 against 67.45 for the text, for example). We also give the correlation between the part $c_m$ of the fused prediction and the label, which measures the encoder inside the fusion without any retraining.

#figure(
  table(
    columns: 5,
    align: (left, left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Encoder*], [*Trained*], [*Acc-2*], [*F1*], [*Corr*]),
    table.hline(stroke: 0.5pt),
    [Text],  [alone],               [78.43 ± 0.64], [78.46 ± 0.62], [67.69 ± 0.79],
    [Text],  [jointly],             [77.94 ± 0.69], [77.94 ± 0.72], [66.47 ± 0.26],
    [Text],  [jointly, inside fusion], [--], [--], [66.18 ± 0.28],
    table.hline(stroke: 0.3pt),
    [Audio], [alone],               [54.66 ± 0.44], [52.37 ± 0.76], [23.38 ± 1.21],
    [Audio], [jointly],             [54.62 ± 0.47], [52.58 ± 0.80], [22.71 ± 0.80],
    [Audio], [jointly, inside fusion], [--], [--], [22.54 ± 0.56],
    table.hline(stroke: 0.3pt),
    [Video], [alone],               [51.90 ± 1.48], [47.71 ± 2.52], [15.75 ± 2.75],
    [Video], [jointly],             [50.29 ± 1.64], [44.45 ± 4.14], [11.96 ± 2.42],
    [Video], [jointly, inside fusion], [--], [--], [11.36 ± 1.22],
    table.hline(),
  ),
  caption: [Each encoder evaluated alone on the CMU-MOSI test set, with a new linear layer trained on the frozen encoder. "Alone": encoder of the uni-modal model; "jointly": encoder taken from the concatenation fusion. "Inside fusion": correlation between the part $c_m$ of the fused prediction and the label (no retraining). Mean ± standard deviation over 3 seeds.],
) <tab-ogm-probe>

== Interpretation

*The text dominates the fusion.* The fusion is not better than the text alone, and 95% of its prediction comes from the text. This is the same behaviour as LF-DNN and TFN in @tab-mosi, whose TAV and T results are almost identical. Our fusion is also at the same level as these two models in TAV (about 77% of Acc-2 and 66 of correlation).

*The video encoder is under-trained inside the fusion.* Trained inside the fusion, the video encoder is clearly worse than when it is trained alone: its correlation drops from 15.75 to 11.96 and its F1 from 47.71 to 44.45 (@tab-ogm-probe). This is the observation of @peng2022ogm: the joint training does not let the weak modality learn as much as it could. (@sec-ogm-ge confirms the F1 drop with 10 seeds, but not the correlation drop, and shows that the video encoder stays at chance level.) For the audio, the drop is very small (23.38 against 22.71) and smaller than the standard deviations, so we cannot conclude. The text encoder also loses a little (67.69 against 66.47), which is coherent with the fact that the fusion is slightly below the text alone in correlation.

*Why the fusion gains nothing.* Two effects add up. First, as explained above, once the text makes the error small, the gradient that reaches the audio and video encoders is small too, and the linear layer gives them small weights. Second, on CMU-MOSI the audio and video features are poor: alone, they do not beat the majority class. Even if the imbalance were corrected, the potential gain would be limited. On this dataset, the imbalance is therefore real, but the main limit is the weak information in the non-text modalities.

*Limits.* These results use only three seeds, and the differences for the audio are within the noise. Our setting is also different from the one of the paper (regression, pre-extracted features, small encoders, Adam). @sec-ogm-ge applies OGM-GE to this fusion, with 10 seeds.
