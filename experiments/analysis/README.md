# CH-SIMS label analysis

**Question:** where does a text-only model fall short? On MOSI, a zero-shot LLM reading only
transcripts beats all our multimodal models (88.6% vs. 79.8% Acc-2). MOSI can't show where
text alone fails, but CH-SIMS can: besides the overall label, each clip has labels annotated
from the text alone, the audio alone and the video alone.

**Method:** compare the LLM predictions from `experiments/mmsa_baselines/llm_baseline.py`
(zero-shot, transcript only) with the overall label and with the text-only label. Split the
test set into clips where the text and overall labels agree in sign and clips where they
conflict, meaning tone or facial expression change what the words suggest.

CH-SIMS is otherwise not used in the project. This is a side analysis for the Discussion.

## Results

CH-SIMS test set: 457 clips, 75 (16.4%) where the text and overall labels conflict.
Accuracy (Acc-2, %) against the overall label:

| Clips | DeepSeek (deepseek-chat) | Qwen (qwen3.8-27b) |
|---|---|---|
| All (457) | 75.3 | 72.4 |
| Text and overall agree (382) | 84.3 | 81.9 |
| Text and overall conflict (75) | **29.3** | **24.0** |

On the conflicting clips, 70.7% (DeepSeek) and 76.0% (Qwen) of predictions match the
text-only label instead. Against the text-only label overall: 82.1% / 81.0% Acc-2,
correlation 81.0 / 79.2 (vs. 64.0 / 60.7 against the overall label).
Full numbers: `results/sims_label_analysis.csv`.

## What it means

- The LLMs do well where the words carry the meaning and fail where they don't. On
  conflicting clips they score far below chance (50%) because they follow the words.
- This is systematic, not noise: with 75 clips the uncertainty is about ±10 points,
  and the gap is about 55 points.
- **Ceiling for any text-only model:** text and overall labels agree on 382 of 457 clips,
  so even a perfect text reader is capped at 83.6% Acc-2 on CH-SIMS. Anything above that
  must come from audio or video.

**Caution:** the predictions also correlate with the audio-only (54–56) and vision-only
(43–44) labels. This does *not* mean the LLM uses audio or video: the four labels are
correlated with each other, since people usually sound and look the way they talk.

## Draft text for the report (Discussion)

> On MOSI, a zero-shot LLM reading only transcripts outperforms all multimodal models. To see
> where text alone falls short, we use CH-SIMS, whose separate text-only labels reveal clips
> where tone or facial expression change the meaning of the words (16% of the test set).
> There, LLM accuracy drops from 82–84% to 24–29%, below chance, as the models follow the
> words. Even a perfect text-only model would be capped at 83.6%.

**Limitations to state:** one dataset (Chinese), 75 conflicting clips, zero-shot prompting
only, and the per-modality labels come from human annotators, so they contain noise too.

## Reproduce

Needs only numpy and the CH-SIMS feature file from MMSA (`unaligned_39.pkl`, not in the repo).

```bash
python3 experiments/analysis/sims_label_analysis.py <path>/unaligned_39.pkl ~/results/llm/sims_*_predictions.csv
```

It writes `results/sims_label_analysis.csv` (numbers only) and
`~/results/llm/sims_conflict_examples.csv`: clips where the LLM followed the words but the
clip meant the opposite, useful for picking an example for the slides. That file contains
CH-SIMS transcripts, so **don't commit it** (the repo is public).
