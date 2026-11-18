"""
Setup
    export LLM_BASE_URL=https://.../v1#openai
    export LLM_API_KEY=sk-...

Usage:
    python llm_baseline.py --dataset mosi --model qwen3.8-27b              

Writes per-clip predictions (resumes if interrupted) and an MMSA-style results CSV
that scripts/add_results.py can read (add with --source llm).
"""
import argparse, csv, json, os, pickle, re, time, urllib.request
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

DATA = {"mosi": ("unaligned_50.pkl", 3), "sims": ("unaligned_39.pkl", 1)}  # file, scale max
PROMPT = ("Rate the sentiment of each utterance (spoken in an opinion video) on a scale "
          "from -{s} (very negative) to +{s} (very positive); decimals allowed. "
          "Reply with only a JSON list of numbers, one per utterance, in the same order.\n\n{items}")


def ask(model, texts, scale):
    items = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT.format(s=scale, items=items)}]}
    req = urllib.request.Request(
        os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["LLM_API_KEY"]})
    with urllib.request.urlopen(req, timeout=120) as resp:
        reply = json.load(resp)["choices"][0]["message"]["content"]
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)  # drop reasoning, if any
    scores = json.loads(reply[reply.index("["): reply.rindex("]") + 1])
    if len(scores) != len(texts):
        raise ValueError(f"expected {len(texts)} scores, got {len(scores)}")
    return [max(-scale, min(scale, float(x))) for x in scores]


def ask_with_retry(model, texts, scale, wait):
    for attempt in range(4):
        try:
            return ask(model, texts, scale)
        except Exception as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(wait * (attempt + 2))
    if len(texts) > 1:  # batch keeps failing: try clips one by one
        return [s for t in texts for s in ask_with_retry(model, [t], scale, wait)]
    raise RuntimeError("giving up on a clip")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mosi", choices=DATA)
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--data-dir", default="~/data/mmsa")
    p.add_argument("--out-dir", default="~/results/llm")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--wait", type=float, default=6.5, help="seconds between requests (limit 10/min)")
    p.add_argument("--limit", type=int, default=0, help="only the first N test clips")
    args = p.parse_args()

    fname, scale = DATA[args.dataset]
    test = pickle.load(open(Path(args.data_dir, fname).expanduser(), "rb"))["test"]
    ids, texts = list(test["id"]), list(test["raw_text"])
    labels = np.asarray(test["regression_labels"], dtype=float).reshape(-1)
    if args.limit:
        ids, texts, labels = ids[:args.limit], texts[:args.limit], labels[:args.limit]

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    pred_file = out / f"{args.dataset}_{args.model}_predictions.csv"
    done = {r["id"]: float(r["pred"]) for r in csv.DictReader(open(pred_file))} if pred_file.exists() else {}

    todo = [i for i, clip in enumerate(ids) if clip not in done]
    print(f"{len(done)} clips already done, {len(todo)} to go")
    with open(pred_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not done:
            writer.writerow(["id", "label", "pred", "text"])
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            scores = ask_with_retry(args.model, [texts[i] for i in batch], scale, args.wait)
            for i, s in zip(batch, scores):
                writer.writerow([ids[i], labels[i], s, texts[i]])
                done[ids[i]] = s
            f.flush()
            print(f"  {len(done)}/{len(ids)}")
            time.sleep(args.wait)

    pred = np.array([done[c] for c in ids])
    # Binary split as in MMSA: MOSI counts 0 as positive, CH-SIMS counts 0 as negative
    pos_true = labels >= 0 if args.dataset == "mosi" else labels > 0
    pos_pred = pred >= 0 if args.dataset == "mosi" else pred > 0
    acc2 = 100 * np.mean(pos_true == pos_pred)
    f1 = 100 * f1_score(pos_true, pos_pred, average="weighted")
    mae = 100 * np.mean(np.abs(pred - labels))
    corr = 100 * np.corrcoef(pred, labels)[0, 1]
    print(f"\n{args.model} on {args.dataset} ({len(ids)} clips): "
          f"Acc-2 {acc2:.2f}  F1 {f1:.2f}  MAE {mae:.2f}  Corr {corr:.2f}")

    res_file = out / f"{args.dataset}_{args.model}_results.csv"
    with open(res_file, "w", newline="") as f:
        csv.writer(f).writerows([["Model", "Variant", "Has0_acc_2", "Has0_F1_score", "MAE", "Corr"],
                                 [args.model, "T", f"{acc2:.2f}", f"{f1:.2f}", f"{mae:.2f}", f"{corr:.2f}"]])
    print(f"Results: {res_file}\nPredictions: {pred_file}")


if __name__ == "__main__":
    main()
