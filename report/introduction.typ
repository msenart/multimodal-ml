= Introduction

== Multimodal machine learning

A _modality_ is a way in which information is produced or perceived: text, sound, images and video are examples. Multimodal machine learning covers the methods that learn from several modalities at once @baltrusaitis2019survey. The starting idea is simple: different modalities carry partly different information. They can therefore complement each other, and a model that combines them should, in principle, do better than a model that uses only one.

Combining modalities raises several challenges. @baltrusaitis2019survey identify five of them:
- *representation*: how to encode each modality, or several modalities together, in a form a model can use;
- *translation*: how to map one modality to another, for example an image to its caption;
- *alignment*: how to match elements across modalities, for example a word and the moment in the video where it is spoken;
- *fusion*: how to combine information from the modalities to make a prediction;
- *co-learning*: how to transfer knowledge from one modality to another, especially when one of them has little data.

Modalities also have very different natures. Text is a sequence of discrete symbols, while audio and video are continuous signals, sampled at different rates. In addition, a modality can be noisy, uninformative or missing. A good multimodal model must therefore also learn how much weight to give to each modality.

== Task: multimodal sentiment analysis

We study multimodal sentiment analysis. Each sample is a short video clip in which a person gives their opinion. The model receives three modalities: the transcript of what is said (T), the audio signal (A) and the images of the face (V). It must predict the intensity of the expressed sentiment as a real number. The task is therefore a regression task.

We use two datasets:
- *CMU-MOSI* @zadeh2016mosi, in English. It contains 2,199 clips (1,284 for training, 229 for validation and 686 for testing). Each clip is labeled with a score in $[-3, 3]$.
- *CH-SIMS* @yu2020chsims, in Chinese. It contains 2,281 clips (1,368, 456 and 457). Each clip is labeled with a score in $[-1, 1]$. In addition to the multimodal score, CH-SIMS provides a separate score for each modality.

In both cases, we use the pre-extracted features provided by the MMSA toolkit @mao2022msena, not the raw videos. For CMU-MOSI, each clip is described by 768-dimensional BERT embeddings for the text, 5-dimensional vectors for the audio and 20-dimensional vectors for the video. These sequences are not aligned in time (_unaligned_ version).
