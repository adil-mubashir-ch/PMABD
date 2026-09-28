"""
build_benchmark_sheet.py
========================
Emit the TinyImageNet-200 benchmark comparison as CSV + XLSX:
published SOTA numbers alongside our measured PMABD ladder.

Published numbers are transcribed from the papers' own tables (table number
recorded per row in the Notes column). Our numbers are read live from
<output_dir>/logs/stages.json so the sheet cannot drift from the run.

    python build_benchmark_sheet.py
    python build_benchmark_sheet.py --config <cfg> --outdir results
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import yaml

# ── Sources ─────────────────────────────────────────────────────────────────
SRC = {
    "SQAKD":  "https://arxiv.org/abs/2403.11106",
    "DAQAKD": "https://arxiv.org/abs/2509.03850",
    "FAQD":   "https://arxiv.org/abs/2302.10899",
    "CA-MKD": "https://arxiv.org/abs/2201.00007",
    "MMKD":   "https://arxiv.org/abs/2306.06634",
    "AMTML":  "https://arxiv.org/abs/2103.04062",
    "OURS":   "",
}
VENUE = {
    "SQAKD":  ("AISTATS", 2024),
    "DAQAKD": ("arXiv preprint", 2025),
    "FAQD":   ("IEEE Access", 2023),
    "CA-MKD": ("ICASSP", 2022),
    "MMKD":   ("ICME", 2023),
    "AMTML":  ("AAAI", 2020),
}

COLUMNS = [
    "Group", "Dataset", "Student Model", "Teacher(s)", "Precision", "Method",
    "Best Val Top-1", "Test Top-1", "Test Top-5", "FP32 Ref Top-1",
    "Delta vs FP32 (pp)", "Retention (%)", "Epochs", "Status",
    "Venue", "Year", "Source", "Notes",
]

# Published MobileNetV2 FP32 references on TinyImageNet
MBV2_FP_SQAKD = 58.07
MBV2_FP_DAQAKD = 58.64
R18_FP_SQAKD = 65.59
R18_FP_DAQAKD = 66.87
VGG11_FP_SQAKD = 59.47
SHUFFLE_FP = 49.91
SQUEEZE_FP = 51.49


def row(group, model, teachers, prec, method, val, t1, t5, fp_ref, epochs,
        status, key, notes):
    venue, year = VENUE.get(key, ("", ""))
    return {
        "Group": group, "Dataset": "TinyImageNet-200", "Student Model": model,
        "Teacher(s)": teachers, "Precision": prec, "Method": method,
        "Best Val Top-1": val, "Test Top-1": t1, "Test Top-5": t5,
        "FP32 Ref Top-1": fp_ref, "Delta vs FP32 (pp)": "",
        "Retention (%)": "", "Epochs": epochs, "Status": status,
        "Venue": venue, "Year": year, "Source": SRC.get(key, ""), "Notes": notes,
    }


def published_rows():
    R = []
    T = "MobileNetV2"

    # ── MobileNetV2 — the head-to-head comparison for our student ──────────
    R.append(row("SQAKD", T, "self / none", "FP32", "Full precision",
                 "", MBV2_FP_SQAKD, 80.97, MBV2_FP_SQAKD, "", "published", "SQAKD",
                 "SQAKD Table 6. 64px upsampled to 224, ImageNet-pretrained init."))
    R.append(row("SQAKD (baseline QAT)", T, "n/a", "W8A8", "DoReFa (QAT only)",
                 "", 56.26, 79.64, MBV2_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 6"))
    R.append(row("SQAKD", T, "FP32 self", "W8A8", "SQAKD (DoReFa)",
                 "", 58.13, 81.30, MBV2_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 6"))
    R.append(row("SQAKD (baseline QAT)", T, "n/a", "W4A4", "PACT (QAT only)",
                 "", 50.33, 75.08, MBV2_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 6"))
    R.append(row("SQAKD", T, "FP32 self", "W4A4", "SQAKD (PACT)",
                 "", 57.14, 80.61, MBV2_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 6"))
    R.append(row("SQAKD (baseline QAT)", T, "n/a", "W3A3", "PACT (QAT only)",
                 "", 47.77, 73.44, MBV2_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 6"))
    R.append(row("SQAKD", T, "FP32 self", "W3A3", "SQAKD (PACT)",
                 "", 52.73, 77.68, MBV2_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 6"))

    R.append(row("DAQAKD", T, "FP32 self", "FP32", "Full precision",
                 "", MBV2_FP_DAQAKD, "", MBV2_FP_DAQAKD, "", "published", "DAQAKD",
                 "DAQAKD Table 3 (their FP baseline)"))
    R.append(row("DAQAKD", T, "FP32 self", "W4A4", "DAQAKD (PACT)",
                 "", 59.48, "", MBV2_FP_DAQAKD, "", "published", "DAQAKD",
                 "DAQAKD Table 3 - best published MobileNetV2 W4A4"))
    R.append(row("DAQAKD", T, "FP32 self", "W3A3", "DAQAKD (PACT)",
                 "", 56.17, "", MBV2_FP_DAQAKD, "", "published", "DAQAKD",
                 "DAQAKD Table 3 - best published MobileNetV2 W3A3"))

    R.append(row("(gap in the literature)", T, "-", "W2A2", "no published result",
                 "", "", "", "", "", "OPEN", "OURS",
                 "No published MobileNetV2 W2A2 on TinyImageNet found. This is our claim."))

    # ── ResNet-18 — the other arch both papers report ──────────────────────
    T = "ResNet-18"
    for prec, pact, lsq, dorefa in [
        ("W8A8", 64.91, 65.08, 63.23),
        ("W4A4", 61.06, 64.10, 62.72),
        ("W3A3", 58.09, 61.99, 61.94),
    ]:
        for name, v in [("PACT (QAT only)", pact), ("LSQ (QAT only)", lsq),
                        ("DoReFa (QAT only)", dorefa)]:
            R.append(row("SQAKD (baseline QAT)", T, "n/a", prec, name, "", v, "",
                         R18_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 5"))
    R.append(row("SQAKD (baseline QAT)", T, "n/a", "FP32", "Full precision", "",
                 R18_FP_SQAKD, "", R18_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 5"))
    for prec, pact, lsq, dorefa in [
        ("W8A8", 65.78, 65.96, 64.88),
        ("W4A4", 61.47, 65.34, 64.56),
        ("W3A3", 61.34, 65.21, 64.10),
    ]:
        for name, v in [("SQAKD (PACT)", pact), ("SQAKD (LSQ)", lsq),
                        ("SQAKD (DoReFa)", dorefa)]:
            R.append(row("SQAKD", T, "FP32 self", prec, name, "", v, "",
                         R18_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 5"))
    R.append(row("DAQAKD", T, "FP32 self", "FP32", "Full precision", "",
                 R18_FP_DAQAKD, "", R18_FP_DAQAKD, "", "published", "DAQAKD", "DAQAKD Table 3"))
    for prec, method, v in [("W8A8", "DAQAKD (DoReFa)", 67.19),
                            ("W4A4", "DAQAKD (PACT)", 67.39),
                            ("W3A3", "DAQAKD (PACT)", 62.31),
                            ("W3A3", "DAQAKD (LSQ)", 66.47)]:
        R.append(row("DAQAKD", T, "FP32 self", prec, method, "", v, "",
                     R18_FP_DAQAKD, "", "published", "DAQAKD", "DAQAKD Table 3"))
    R.append(row("FAQD", T, "FP32 self", "FP32", "Full precision", "", 64.23, "",
                 64.23, "", "published", "FAQD",
                 "FAQD also upsamples 64px to 224 - third confirmation of that convention"))

    # ── VGG-11 (SQAKD Table 5) ────────────────────────────────────────────
    T = "VGG-11"
    R.append(row("SQAKD (baseline QAT)", T, "n/a", "FP32", "Full precision", "",
                 VGG11_FP_SQAKD, "", VGG11_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 5"))
    for prec, pact, lsq, dorefa in [
        ("W8A8", 58.08, 59.25, 57.54),
        ("W4A4", 57.10, 59.14, 57.28),
        ("W3A3", 52.94, 58.39, 56.72),
    ]:
        for name, v in [("PACT (QAT only)", pact), ("LSQ (QAT only)", lsq),
                        ("DoReFa (QAT only)", dorefa)]:
            R.append(row("SQAKD (baseline QAT)", T, "n/a", prec, name, "", v, "",
                         VGG11_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 5"))
    for prec, pact, lsq, dorefa in [
        ("W8A8", 59.44, 59.42, 58.91),
        ("W4A4", 59.05, 59.19, 58.93),
        ("W3A3", 57.25, 58.43, 57.02),
    ]:
        for name, v in [("SQAKD (PACT)", pact), ("SQAKD (LSQ)", lsq),
                        ("SQAKD (DoReFa)", dorefa)]:
            R.append(row("SQAKD", T, "FP32 self", prec, name, "", v, "",
                         VGG11_FP_SQAKD, "", "published", "SQAKD", "SQAKD Table 5"))

    # ── Other lightweight archs (SQAKD Table 6) ───────────────────────────
    for T, fp, fp5, entries in [
        ("ShuffleNet-V2", SHUFFLE_FP, 76.05, [
            ("W4A4", "PACT (QAT only)", 27.09, 52.54, "SQAKD (baseline QAT)"),
            ("W4A4", "SQAKD (PACT)", 41.11, 68.40, "SQAKD"),
            ("W8A8", "DoReFa (QAT only)", 45.96, 71.93, "SQAKD (baseline QAT)"),
            ("W8A8", "SQAKD (DoReFa)", 47.33, 73.85, "SQAKD")]),
        ("SqueezeNet", SQUEEZE_FP, 76.02, [
            ("W4A4", "LSQ (QAT only)", 35.37, 62.75, "SQAKD (baseline QAT)"),
            ("W4A4", "SQAKD (LSQ)", 47.40, 73.18, "SQAKD"),
            ("W8A8", "DoReFa (QAT only)", 42.66, 69.25, "SQAKD (baseline QAT)"),
            ("W8A8", "SQAKD (DoReFa)", 46.62, 73.02, "SQAKD")]),
    ]:
        R.append(row("SQAKD (baseline QAT)", T, "n/a", "FP32", "Full precision", "",
                     fp, fp5, fp, "", "published", "SQAKD", "SQAKD Table 6"))
        for prec, method, t1, t5, grp in entries:
            R.append(row(grp, T, "FP32 self" if "SQAKD (" in method else "n/a",
                         prec, method, "", t1, t5, fp, "", "published", "SQAKD",
                         "SQAKD Table 6"))

    # ── Multi-teacher KD (FP32 only, CIFAR-style archs at 64px) ───────────
    note_mt = ("DIFFERENT SETTING: CIFAR-style architectures at native 64px, "
               "not comparable to the 224px rows above. Cite for positioning only.")
    for method, v in [("Student (no KD)", 44.40), ("KD", 47.42), ("FitNet", 47.24),
                      ("AT", 45.73), ("VID", 47.76), ("CRD", 48.11), ("CA-MKD", 49.55)]:
        R.append(row("Multi-teacher KD", "VGG8", "ResNet32x4 (53.38)", "FP32", method,
                     "", v, "", 44.40, "", "published", "CA-MKD",
                     "CA-MKD Table 6. " + note_mt))
    for method, v in [("Student (no KD)", 39.46), ("AVER", 41.87), ("FitNet-MKD", 41.46),
                      ("EBKD", 41.46), ("AEKD-logits", 41.19), ("AEKD-feature", 41.56),
                      ("CA-MKD", 42.65), ("MMKD", 44.15)]:
        R.append(row("Multi-teacher KD", "MobileNetV2 (CIFAR-style)", "3x VGG13", "FP32",
                     method, "", v, "", 39.46, "", "published", "MMKD",
                     "MMKD Table III. " + note_mt))
    for method, v in [("FitNet", 54.85), ("RKD", 54.94), ("AMTML-KD", 55.67)]:
        R.append(row("Multi-teacher KD", "ResNet-style student", "3x ResNet", "FP32",
                     method, "", v, "", "", "", "published", "AMTML",
                     "AMTML-KD. " + note_mt))
    return R


STAGE_DISPLAY_NAMES = {
    "m3_kd": "M1+M2+M3->MBv2_FP32 (KD warm-up)",
    "m4":    "M1+M2+M3+FP32->M4 (W8A8)",
    "m5a":   "M4->M5a (W8A4)",
    "m5b":   "M5a->M5b (W4A4)",
    "m6a":   "M5b->M6a (W4A3)",
    "m6b":   "M6a->M6b (W3A3)",
    "m7a":   "M6b->M7a (W3A2)",
    "m7b":   "M7a->M7b (W2A2)",
}


def in_progress(epochs_csv, stage_id):
    """
    Best-so-far for a stage that has not finished yet.

    stages.json is only written when a stage completes, so a running stage is
    invisible there — but every epoch is already in epochs.csv.
    """
    name = STAGE_DISPLAY_NAMES.get(stage_id)
    if not name or not os.path.isfile(epochs_csv):
        return None
    rows = [r for r in csv.DictReader(open(epochs_csv, encoding="utf-8"))
            if r.get("stage") == name and r.get("val_top1")]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r["val_top1"]))
    return {
        "best_val_top1": round(float(best["val_top1"]), 2),
        "test_top1": round(float(best["test_top1"]), 2),
        "test_top5": round(float(best["test_top5"]), 2),
        "epochs_run": len(rows),
    }


def our_rows(stages_path):
    """Read our measured results straight out of the run's stages.json."""
    if not os.path.isfile(stages_path):
        return [], None
    entries = {e["stage_id"]: e for e in json.load(open(stages_path, encoding="utf-8"))}
    epochs_csv = os.path.join(os.path.dirname(stages_path), "epochs.csv")

    fp_ref = entries.get("m3_kd", {}).get("test_top1", "")
    teachers = "ResNet-50 + ResNet-34 + ResNet-18"
    R = []

    for sid, model, label, note in [
        ("pretrain_resnet50", "ResNet-50", "Fine-tuned teacher M1",
         "Teacher, not a result. Peaked at epoch 1 (lr 0.01 too hot); see notes."),
        ("pretrain_resnet34", "ResNet-34", "Fine-tuned teacher M2", "Teacher, not a result."),
        ("pretrain_resnet18", "ResNet-18", "Fine-tuned teacher M3",
         "Teacher. Beats published ResNet-18 FP32 (65.59 / 66.87)."),
        ("pretrain_mobilenet_v2", "MobileNetV2", "Fine-tuned student init (no KD)",
         "Our FP32 student before KD. Far above published 58.07 / 58.64."),
    ]:
        e = entries.get(sid)
        if not e:
            continue
        R.append(row("OURS - teachers / init", model, "n/a", "FP32", label,
                     e.get("best_val_top1", ""), e.get("test_top1", ""),
                     e.get("test_top5", ""), "", e.get("epochs_run", ""),
                     "measured", "OURS", note))

    ladder = [
        ("m3_kd", "FP32", "PMABD (KD warm-up)"),
        ("m4",    "W8A8", "PMABD"),
        ("m5a",   "W8A4", "PMABD"),
        ("m5b",   "W4A4", "PMABD"),
        ("m6a",   "W4A3", "PMABD"),
        ("m6b",   "W3A3", "PMABD"),
        ("m7a",   "W3A2", "PMABD"),
        ("m7b",   "W2A2", "PMABD"),
    ]
    for sid, prec, method in ladder:
        e = entries.get(sid)
        if e:
            R.append(row("OURS - PMABD ladder", "MobileNetV2", teachers, prec, method,
                         e.get("best_val_top1", ""), e.get("test_top1", ""),
                         e.get("test_top5", ""), fp_ref, e.get("epochs_run", ""),
                         "measured", "OURS",
                         "First/last layer held at W8A8 (SQAKD/DAQAKD leave them FP32 - "
                         "ours is the harder setting)."))
            continue
        live = in_progress(epochs_csv, sid)
        if live:
            R.append(row("OURS - PMABD ladder", "MobileNetV2", teachers, prec, method,
                         live["best_val_top1"], live["test_top1"], live["test_top5"],
                         fp_ref, live["epochs_run"], "IN PROGRESS", "OURS",
                         f"Best of {live['epochs_run']} epochs so far - not final."))
        else:
            R.append(row("OURS - PMABD ladder", "MobileNetV2", teachers, prec, method,
                         "", "", "", fp_ref, "", "PENDING", "OURS", "Not yet run."))
    return R, fp_ref


# ─────────────────────────────────────────────────────────────────────────────
# XLSX
# ─────────────────────────────────────────────────────────────────────────────

ACC_FMT = '0.00"%"'
PP_FMT = '+0.00;-0.00;0.00'

# MobileNetV2 head-to-head: the table that goes in the paper.
HEAD_PRECISIONS = ["FP32", "W8A8", "W4A4", "W3A3", "W2A2"]


def _style_header(ws, ncols, row_idx=1):
    from openpyxl.styles import Alignment, Font, PatternFill
    fill = PatternFill("solid", fgColor="1F3864")
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row_idx, column=c)
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def write_xlsx(rows, fp_ref, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    # ── Sheet 1: full benchmark ──────────────────────────────────────────
    ws = wb.active
    ws.title = "Benchmark"
    ws.append(COLUMNS)
    _style_header(ws, len(COLUMNS))

    t1c = get_column_letter(COLUMNS.index("Test Top-1") + 1)
    fpc = get_column_letter(COLUMNS.index("FP32 Ref Top-1") + 1)
    di = COLUMNS.index("Delta vs FP32 (pp)") + 1
    ri = COLUMNS.index("Retention (%)") + 1
    src_i = COLUMNS.index("Source") + 1

    ours_fill = PatternFill("solid", fgColor="FFF2CC")
    open_fill = PatternFill("solid", fgColor="D9EAD3")
    mt_fill = PatternFill("solid", fgColor="F2F2F2")

    for r in rows:
        excel_row = ws.max_row + 1
        ws.append([r[c] for c in COLUMNS])
        if r["Test Top-1"] != "" and r["FP32 Ref Top-1"] != "":
            ws.cell(row=excel_row, column=di).value = (
                f"=IFERROR({t1c}{excel_row}-{fpc}{excel_row},\"\")")
            ws.cell(row=excel_row, column=ri).value = (
                f"=IFERROR({t1c}{excel_row}/{fpc}{excel_row}*100,\"\")")
        fill = None
        if r["Group"].startswith("OURS"):
            fill = ours_fill
        elif r["Status"] == "OPEN":
            fill = open_fill
        elif r["Group"] == "Multi-teacher KD":
            fill = mt_fill
        for c in range(1, len(COLUMNS) + 1):
            cell = ws.cell(row=excel_row, column=c)
            cell.font = Font(name="Arial", size=10,
                             bold=r["Group"].startswith("OURS"))
            if fill:
                cell.fill = fill
        for name in ("Best Val Top-1", "Test Top-1", "Test Top-5",
                     "FP32 Ref Top-1", "Retention (%)"):
            ws.cell(row=excel_row, column=COLUMNS.index(name) + 1).number_format = ACC_FMT
        ws.cell(row=excel_row, column=di).number_format = PP_FMT
        if r["Source"]:
            cell = ws.cell(row=excel_row, column=src_i)
            cell.hyperlink = r["Source"]
            cell.font = Font(name="Arial", size=10, color="0563C1", underline="single")

    widths = {"Group": 24, "Dataset": 17, "Student Model": 24, "Teacher(s)": 30,
              "Precision": 10, "Method": 26, "Notes": 70, "Source": 34,
              "Venue": 14, "Status": 12}
    for i, name in enumerate(COLUMNS, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(name, 13)
    ws.freeze_panes = "G2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{ws.max_row}"

    # ── Sheet 2: MobileNetV2 head-to-head ────────────────────────────────
    hs = wb.create_sheet("Head-to-head (MobileNetV2)")
    hdr = ["Precision", "PMABD (ours)", "DAQAKD 2025", "SQAKD AISTATS'24",
           "Best QAT-only", "Ours - best published (pp)"]
    hs.append(hdr)
    _style_header(hs, len(hdr))

    ours_by_prec, mbv2 = {}, {}
    for r in rows:
        if r["Group"] == "OURS - PMABD ladder" and r["Test Top-1"] != "":
            ours_by_prec[r["Precision"]] = r["Test Top-1"]
        if r["Student Model"] == "MobileNetV2" and r["Test Top-1"] != "":
            grp = ("DAQAKD" if r["Group"] == "DAQAKD"
                   else "SQAKD" if r["Group"] == "SQAKD"
                   else "QAT" if r["Group"] == "SQAKD (baseline QAT)" else None)
            if grp:
                key = (r["Precision"], grp)
                mbv2[key] = max(mbv2.get(key, 0), r["Test Top-1"])

    for prec in HEAD_PRECISIONS:
        excel_row = hs.max_row + 1
        hs.append([prec, ours_by_prec.get(prec, ""),
                   mbv2.get((prec, "DAQAKD"), ""), mbv2.get((prec, "SQAKD"), ""),
                   "" if prec == "FP32" else mbv2.get((prec, "QAT"), ""), ""])
        best_cells = [f"C{excel_row}", f"D{excel_row}", f"E{excel_row}"]
        hs.cell(row=excel_row, column=6).value = (
            f'=IF(OR(B{excel_row}="",COUNT({",".join(best_cells)})=0),"",'
            f'B{excel_row}-MAX({",".join(best_cells)}))')
        for c in range(1, 7):
            cell = hs.cell(row=excel_row, column=c)
            cell.font = Font(name="Arial", size=10, bold=(c in (1, 2)))
            if c in (2, 6):
                cell.fill = ours_fill
            if 2 <= c <= 5:
                cell.number_format = ACC_FMT
        hs.cell(row=excel_row, column=6).number_format = PP_FMT

    note_row = hs.max_row + 2
    for i, line in enumerate([
        "Ours = MobileNetV2 student, teachers ResNet-50 + ResNet-34 + ResNet-18, "
        "TinyImageNet 64px upsampled to 224, ImageNet-pretrained init.",
        "Blank cells = no published number at that precision. W2A2 is unreported "
        "in every paper surveyed - that column is the contribution.",
        "CAVEAT: our FP32 baseline is far stronger than theirs "
        f"(ours {fp_ref} vs SQAKD 58.07 / DAQAKD 58.64), so absolute gaps flatter us "
        "and relative retention flatters them. Report both.",
        "Our first conv and last FC are held at W8A8; SQAKD and DAQAKD leave "
        "theirs at full precision. Ours is the harder setting.",
    ], start=0):
        c = hs.cell(row=note_row + i, column=1, value=line)
        c.font = Font(name="Arial", size=9, italic=True)
    for col, w in zip("ABCDEF", [12, 16, 15, 19, 16, 26]):
        hs.column_dimensions[col].width = w
    hs.freeze_panes = "A2"

    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mobilenetv2_tinyimagenet_2bit_ladder.yaml")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    stages = os.path.join(cfg["experiment"]["output_dir"], "logs", "stages.json")

    ours, fp_ref = our_rows(stages)
    rows = ours + published_rows()
    os.makedirs(args.outdir, exist_ok=True)

    # ── CSV (formulas for the two derived columns; Sheets evaluates them) ──
    csv_path = os.path.join(args.outdir, "tinyimagenet_benchmark.csv")
    di = COLUMNS.index("Delta vs FP32 (pp)")
    ri = COLUMNS.index("Retention (%)")
    t1c = chr(ord("A") + COLUMNS.index("Test Top-1"))
    fpc = chr(ord("A") + COLUMNS.index("FP32 Ref Top-1"))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for i, r in enumerate(rows, start=2):
            vals = [r[c] for c in COLUMNS]
            if r["Test Top-1"] != "" and r["FP32 Ref Top-1"] != "":
                vals[di] = f"=IFERROR({t1c}{i}-{fpc}{i},\"\")"
                vals[ri] = f"=IFERROR({t1c}{i}/{fpc}{i}*100,\"\")"
            w.writerow(vals)
    print(f"CSV  -> {csv_path}  ({len(rows)} rows)")

    xlsx_path = os.path.join(args.outdir, "tinyimagenet_benchmark.xlsx")
    write_xlsx(rows, fp_ref, xlsx_path)
    print(f"XLSX -> {xlsx_path}")
    return rows, fp_ref, args.outdir


if __name__ == "__main__":
    main()
