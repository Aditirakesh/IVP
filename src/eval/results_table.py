"""results.csv (+ efficiency.csv) -> reports/results_table.md

    python -m src.eval.results_table --results runs/results.csv --eff reports/efficiency.csv

results.csv columns: experiment, protocol, level, metric, mean, sd  (values are fractions, 0 to 1)
Names this looks for:
    experiment  cdl_mddnet (full 128 ch) and cdl_mddnet_ch<N> (reduced montage, N = channel count)
    protocol    kfold10, loso
    level       participant (majority vote) or epoch
Targets are the ones on deck slides 39 and 40.
"""
import argparse
import csv
import os
import re

ACC10, ACCLOSO, AUC, PARAMS, DROP = 0.85, 0.80, 0.88, 100_000, 0.03


def readCsv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get(res, exp, proto, metric, level="participant"):
    for r in res:
        if (r["experiment"], r["protocol"], r["level"], r["metric"]) == (exp, proto, level, metric):
            return float(r["mean"]), float(r["sd"] or 0)
    return None


def pct(v):
    # LOSO is pooled over all subjects so it has no sd, leave it blank in the csv
    s = f"{v[0] * 100:.1f}%"
    return s + (f" ± {v[1] * 100:.1f}" if v[1] else "")


def measures(res, eff):
    """one dict per row of the deck table (slide 39), same order and names"""
    out = []

    def add(name, pub, target, got, ok, short=None):
        # short = the compact form the crowded slide 40 uses
        out.append({"name": name, "pub": pub, "target": target,
                    "got": got or "not yet measured", "short": short or got or "",
                    "status": "pending" if ok is None else ("met" if ok else "not met")})

    v = get(res, "cdl_mddnet", "kfold10", "accuracy")
    add("MODMA 10-fold, cross-subject", "99.33% [1]", "≥ 85%", v and pct(v), v and v[0] >= ACC10)

    v = get(res, "cdl_mddnet", "loso", "accuracy")
    add("MODMA LOSO", "99.32% [5]", "≥ 80%", v and pct(v), v and v[0] >= ACCLOSO)

    v = get(res, "cdl_mddnet", "kfold10", "auc")
    add("AUC (subject-independent)", "99.96% [1]", "≥ 0.88",
        v and f"{v[0]:.3f}" + (f" ± {v[1]:.3f}" if v[1] else ""), v and v[0] >= AUC)

    n = None
    for r in eff:
        if "cdl" in r["model"] and r["channels"] == "128":
            n = int(r["params"])
    add("Parameters", "34,812 [1]", "< 100k", n and f"{n:,}", n and n < PARAMS)

    # best of the reduced sets with 32 channels or fewer, against the full montage
    full = get(res, "cdl_mddnet", "kfold10", "accuracy")
    small = []
    for r in res:
        m = re.fullmatch(r"cdl_mddnet_ch(\d+)", r["experiment"])
        key = (r["protocol"], r["level"], r["metric"])
        if m and int(m.group(1)) <= 32 and key == ("kfold10", "participant", "accuracy"):
            small.append((float(r["mean"]), int(m.group(1))))
    got = short = ok = None
    if full and small:
        acc, ch = max(small)
        d = acc - full[0]
        got = f"{ch} ch: {acc * 100:.1f}%, {d * 100:+.1f} points vs full"
        short = f"{ch} ch, {d * 100:+.1f} points"
        ok = d >= -DROP
    add("Channels", "16 of 128 [6]", "≤ 32, within 3 points of full montage", got, ok, short)

    return out


def table(ms):
    L = ["| Measure | Published (literature) | Target | CDL-MDDNet (measured) | Status |",
         "| --- | --- | --- | --- | --- |"]
    for m in ms:
        L.append(f"| {m['name']} | {m['pub']} | {m['target']} | {m['got']} | {m['status']} |")
    return "\n".join(L)


def allMetrics(res):
    cols, rows = [], {}
    for r in res:
        if r["metric"] not in cols:
            cols.append(r["metric"])
        k = (r["experiment"], r["protocol"], r["level"])
        rows.setdefault(k, {})[r["metric"]] = f"{float(r['mean']):.3f} ± {float(r['sd'] or 0):.3f}"
    if not rows:
        return "No results yet."
    L = ["| Experiment | Protocol | Level | " + " | ".join(cols) + " |",
         "| --- | --- | --- | " + " | ".join("---" for _ in cols) + " |"]
    for k, v in rows.items():
        L.append("| " + " | ".join(k) + " | " + " | ".join(v.get(c, "") for c in cols) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/results.csv")
    ap.add_argument("--eff", default="reports/efficiency.csv")
    ap.add_argument("--out", default="reports/results_table.md")
    a = ap.parse_args()

    res, eff = readCsv(a.results), readCsv(a.eff)
    txt = "# Results\n\n"
    txt += table(measures(res, eff)) + "\n\n"
    txt += ("Published numbers are as reported in the papers. Their protocols differ "
            "(segment split vs LOSO), so equal accuracy does not mean equal performance.\n\n")
    txt += "## All metrics (mean ± SD over folds)\n\n" + allMetrics(res) + "\n"

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
