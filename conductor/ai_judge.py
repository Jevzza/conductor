"""AI judge (RLAIF-style): an audio-language model rates transitions; we only trust it if it
agrees with the user's own ratings.   .venv-ml\\Scripts\\python -m conductor.ai_judge <calibration.json>"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
PROMPT = ("This is a 12-second excerpt of game background music. At the 6-second mark the music changes: "
          "it either continues naturally or cuts/crossfades to another part or another song. As a music producer, "
          "rate how musical and seamless that change sounds: 1 = jarring, like a skipping record or a clumsy edit; "
          "3 = acceptable; 5 = seamless, sounds intended. Answer with only the number.")


def main(manifest: str):
    import librosa
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    items = json.load(open(manifest, encoding="utf-8"))
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    sr = proc.feature_extractor.sampling_rate
    from . import lab
    import os
    digit_ids = [proc.tokenizer.convert_tokens_to_ids(str(k)) for k in range(1, 6)]
    lab.reset("judge", model=MODEL, total=len(items), rows=[])
    scores = []
    for i, it in enumerate(items):
        y, _ = librosa.load(it["file"], sr=sr, mono=True)
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": it["file"]}, {"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[y], sampling_rate=sr, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = model(**inp).logits[0, -1].float()
        probs = torch.softmax(logits[digit_ids], 0).cpu().numpy()      # its confidence in 1..5
        s = float((probs * np.arange(1, 6)).sum())                      # expected rating
        ans = str(int(np.argmax(probs)) + 1)
        scores.append(s)
        lab.append("judge", "rows", {"id": it["id"], "round": it["round"], "kind": it["kind"], "verdict": it["verdict"],
                                     "note": it.get("note", ""), "file": "clips/" + os.path.basename(it["file"]),
                                     "score": round(s, 3), "probs": [round(float(x), 3) for x in probs]})
        if i % 20 == 0:
            print(f"[{i}/{len(items)}] {it['verdict']:>4} → judge {s}  ({ans.strip()[:20]})", flush=True)
    user = np.array([{"bad": 0, "ok": 1, "good": 2}[it["verdict"]] for it in items])
    sc = np.array(scores)
    lab.update("judge", done=True)
    from scipy.stats import spearmanr
    rho = spearmanr(sc, user).correlation
    lab.update("judge", spearman=float(rho))
    good, bad = sc[user == 2], sc[user == 0]
    auc = np.mean([[1.0 if g > b else 0.5 if g == b else 0.0 for b in bad] for g in good]) if len(good) and len(bad) else float("nan")
    print(f"AI judge vs user: Spearman {rho:.2f}, good-vs-bad AUC {auc:.2f}, mean score good {good.mean():.2f} / ok "
          f"{sc[user == 1].mean():.2f} / bad {bad.mean():.2f}")
    lab.update("judge", auc=float(auc))
    json.dump({"scores": scores, "spearman": rho, "auc": auc}, open(manifest.replace(".json", "_aijudge.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1])
