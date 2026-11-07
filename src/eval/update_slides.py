"""Fill measured numbers into the deck (slides 39 and 40).

    python -m src.eval.update_slides deck.pptx --results runs/results.csv --eff reports/efficiency.csv --out deck_measured.pptx

Needs python-pptx. Only text is changed, so fonts and layout stay. Rows that are
still not measured are left exactly as they were. Slides are found by title.
"""
import argparse

from pptx import Presentation

from src.eval.results_table import measures, readCsv

T39 = "Research Gap and Planned Evaluation"
T40 = "Evaluation Metrics & Expected Outcomes"
# slide 40 bullets -> index in measures(): 10-fold, LOSO, AUC, channels
BULLETS = [0, 1, 2, 4]
GROW = 457200  # 0.5 inch in EMU


def findSlide(prs, title):
    for s in prs.slides:
        for sh in s.shapes:
            if sh.has_text_frame and sh.text_frame.text.strip() == title:
                return s
    raise SystemExit(f"slide '{title}' not found, has the deck changed?")


def setText(par, txt):
    # first run keeps the formatting, the rest go
    par.runs[0].text = txt
    for r in par.runs[1:]:
        r._r.getparent().remove(r._r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--results", default="runs/results.csv")
    ap.add_argument("--eff", default="reports/efficiency.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ms = measures(readCsv(a.results), readCsv(a.eff))
    done = [m for m in ms if m["status"] != "pending"]
    if not done:
        raise SystemExit("nothing measured yet, deck not changed")
    prs = Presentation(a.deck)

    # slide 39: table column 3 + the note under it
    s = findSlide(prs, T39)
    for sh in s.shapes:
        if sh.has_table:
            for row in list(sh.table.rows)[1:]:
                name = row.cells[0].text_frame.text.strip()
                for m in done:
                    if m["name"] == name:
                        setText(row.cells[2].text_frame.paragraphs[0],
                                f"{m['got']} (target {m['status']})")
        elif sh.has_text_frame and sh.text_frame.text.startswith("Measured CDL-MDDNet results will"):
            if len(done) == len(ms):
                setText(sh.text_frame.paragraphs[0], "Measured CDL-MDDNet values replace the earlier targets.")

    # slide 40: acceptance targets
    s = findSlide(prs, T40)
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        txt = sh.text_frame.text
        if txt.startswith("Acceptance targets"):
            allmet = len(done) == len(ms) and all(m["status"] == "met" for m in ms)
            setText(sh.text_frame.paragraphs[0], "Acceptance targets (achieved)" if allmet else "Acceptance targets (measured)")
        elif txt.startswith("MODMA 10-fold CV"):
            for par, i in zip(sh.text_frame.paragraphs, BULLETS):
                if ms[i]["status"] != "pending":
                    setText(par, f"{par.text} (measured {ms[i]['short']}, {ms[i]['status']})")
    # the text got longer, so grow the two tinted panels behind it (same height, keeps them level)
    for sh in s.shapes:
        if sh.name in ("Shape 2", "Shape 6"):
            sh.height += GROW

    prs.save(a.out)
    print("saved", a.out, f"({len(done)} of {len(ms)} rows filled)")


if __name__ == "__main__":
    main()
