= Evaluation metrics

The models predict a real score $hat(y)$, which is compared to the annotated score $y$. We first turn this regression problem into a binary classification problem: each score is assigned to the class "positive" or "negative" according to its sign. Let $N$ be the number of test samples.

== Acc-2

Acc-2 (binary accuracy) is the proportion of samples whose predicted class is correct:
$ "Acc-2" = 1/N sum_(i=1)^N bb(1)[c(hat(y)_i) = c(y_i)], $
where $c(dot)$ gives the class of a score and $bb(1)[dot]$ equals 1 if the condition is true and 0 otherwise.

The case of zero scores must be specified, because it changes the value obtained. We follow the conventions of MMSA @mao2022msena:
- on CMU-MOSI, a score $>= 0$ is positive and a score $< 0$ is negative (the so-called _Has0_ variant, which keeps neutral samples);
- on CH-SIMS, a score $> 0$ is positive and a score $<= 0$ is negative.

Another variant common in the literature, called _Non0_, removes the samples whose score is exactly zero. It usually gives higher values. The numbers in this report are therefore only comparable to published numbers that use the same convention.

== F1 score

Acc-2 can be misleading when the classes are imbalanced: a model that always predicts the majority class gets an Acc-2 equal to the proportion of that class, without having learned anything. We therefore also use the F1 score. For a class $k$, it is the harmonic mean of the precision $P_k$ and the recall $R_k$:
$ F_1^((k)) = (2 P_k R_k) / (P_k + R_k). $
We report the _weighted_ F1: the average of the $F_1^((k))$ of the two classes, each weighted by its proportion $n_k \/ N$ in the test set:
$ F_1 = sum_k n_k / N F_1^((k)). $
A model that always predicts the same class gets an F1 of zero on the other class, which strongly lowers the weighted F1.

These reference points help to read the results. On the CMU-MOSI test set, 44.75% of the samples have a score $>= 0$: always predicting "negative" therefore gives an Acc-2 of 55.25% (and an F1 of 39.3). On the CH-SIMS test set, only 30.63% of the samples have a score $> 0$: always predicting "negative" gives an Acc-2 of 69.37% (and an F1 of 56.8).

== Other metrics

The results file also contains two regression metrics: the mean absolute error (MAE, $1/N sum_i |hat(y)_i - y_i|$, lower is better) and the Pearson correlation coefficient between $hat(y)$ and $y$ (Corr, higher is better). As in MMSA, both are multiplied by 100. The MAE depends on the scale of the scores: it cannot be compared between CMU-MOSI ($[-3, 3]$) and CH-SIMS ($[-1, 1]$).
