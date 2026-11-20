#!/usr/bin/env python3
"""Мини-eval для real.py:  python eval_real.py [--backend mock|chaos|groq] [--chaos-mode M] [--brief] [-v]

  groq  — настоящий вызов (нужен GROQ_API_KEY и пакет groq).

Печатает:
  * точность по каждому полю отдельно;
  * галлюцинации (в эталоне null, а поле заполнено) и пропуски (в эталоне значение, а null) — РАЗДЕЛЬНО;
  * «неверное значение» (оба не null, но разные) — третья категория, чтобы цифры сходились;
  * статистику валидации (попытки, провалы) и что сделал аудит;
  * RAW vs FINAL: что модель выдала сама (1-я попытка) и что осталось после валидации и аудита.
Код возврата: 0, если после аудита нет ни одной галлюцинации; 1 — иначе (удобно для CI).
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
from collections import Counter, defaultdict

import real
from checks import canon, warning_code
extract_json = real.extract_json

ROOT = pathlib.Path(__file__).parent
SCORED = ["doc_type", "supplier_name", "supplier_bin", "doc_number", "doc_date",
          "currency", "total_amount", "vat_amount", "line_items"]
AMOUNTS = {"total_amount", "vat_amount"}


def is_null(field, v) -> bool:
    if field == "doc_type":
        return v in (None, "unknown")
    if field == "line_items":
        return not v
    return v is None


def num_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 0.005
    except (TypeError, ValueError):
        return False


def item_eq(p, g) -> bool:
    if not isinstance(p, dict):
        return False
    sim = difflib.SequenceMatcher(None, canon(str(p.get("name", ""))), canon(g["name"])).ratio()
    for k in ("qty", "price"):
        pv, gv = p.get(k), g[k]
        if (pv is None) != (gv is None) or (gv is not None and not num_eq(pv, gv)):
            return False
    return sim >= 0.85


def equal(field, p, g) -> bool:
    if field in AMOUNTS:
        return num_eq(p, g)
    if field in ("supplier_name", "doc_number"):
        return isinstance(p, str) and canon(p) == canon(g)
    if field == "line_items":
        return isinstance(p, list) and len(p) == len(g) and all(item_eq(a, b) for a, b in zip(p, g))
    return p == g


def classify(field, p, g) -> str:
    pn, gn = is_null(field, p), is_null(field, g)
    if pn and gn:
        return "correct"
    if pn:
        return "omission"
    if gn:
        return "hallucination"
    return "correct" if equal(field, p, g) else "wrong"


# Сравниваем только «смысловые» коды. UNGROUNDED_*/RULE_OVERRIDE — это объяснения вмешательств
# аудита (их число зависит от того, как сильно модель ошиблась), в эталон они не входят.
SEMANTIC = {"CONFLICT_DATE", "CONFLICT_TOTAL", "CONFLICT_CURRENCY", "OCR_FIXED", "VAT_EXEMPT"}


def codes(ws) -> set:
    return {warning_code(w) for w in (ws or []) if isinstance(w, str)} & SEMANTIC


def score(preds: dict, golds: dict):
    """-> per_field[field][cls] -> count, warn_ok, per_doc_diffs"""
    per = {f: Counter() for f in SCORED}
    warn_ok, diffs = 0, defaultdict(list)
    for name, g in golds.items():
        p = preds[name]
        for f in SCORED:
            c = classify(f, p.get(f), g[f])
            per[f][c] += 1
            if c != "correct":
                diffs[name].append((f, c, p.get(f), g[f]))
        if codes(p.get("warnings")) == codes(g["warnings"]):
            warn_ok += 1
        else:
            diffs[name].append(("warnings", "wrong", sorted(codes(p.get("warnings"))), sorted(codes(g["warnings"]))))
    return per, warn_ok, diffs


def totals(per):
    t = Counter()
    for f in SCORED:
        t.update(per[f])
    return t


def conf_calibration(finals: dict, golds: dict):
    ok, bad = [], []
    for name, g in golds.items():
        fc = finals[name].get("field_confidence", {})
        for f in SCORED:
            if f in fc and not is_null(f, finals[name].get(f)):
                (ok if classify(f, finals[name].get(f), g[f]) == "correct" else bad).append(fc[f])
    avg = lambda xs: f"{sum(xs) / len(xs):.2f} (n={len(xs)})" if xs else "—"
    return avg(ok), avg(bad)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "chaos", "groq"])
    ap.add_argument("--chaos-mode", default="hallucinate",
                    choices=["clean", "bad_json", "bad_type", "hallucinate", "stubborn"])
    ap.add_argument("--max-attempts", type=int, default=real.MAX_ATTEMPTS)
    ap.add_argument("--brief", action="store_true", help="одна сводка вместо полного отчёта")
    ap.add_argument("-v", "--verbose", action="store_true", help="показать каждое расхождение с эталоном")
    a = ap.parse_args(argv)

    golds = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/gold").glob("*.json"))}
    if a.backend == "groq":
        call = real.groq_call
        call.name = "groq:" + real.MODEL
    backend = call
    finals, raws, stats = {}, {}, Counter()
    unparsable_raw = 0
    for name in golds:
        text = (ROOT / "data/docs" / f"{name}.txt").read_text(encoding="utf-8")
        final, info = real.run_pipeline(text, call, a.max_attempts)
        finals[name] = final.model_dump(mode="json")
        stats["ok" if info.ok else "failed"] += 1
        stats["retried"] += int(info.ok and info.attempts > 1)
        stats["attempts_total"] += info.attempts
        for iv in info.fixes:
            stats["guard:" + iv.split(":")[0]] += 1
        try:
            raws[name] = json.loads(extract_json(info.raw_first or ""))
        except ValueError:
            raws[name] = None
            unparsable_raw += 1

    per, warn_ok, diffs = score(finals, golds)
    n = len(golds)
    if a.brief:
        t = totals(per)
        cells = n * len(SCORED)
        parsed = {k: v for k, v in raws.items() if isinstance(v, dict)}
        rt = totals(score(parsed, {k: golds[k] for k in parsed})[0]) if parsed else Counter()
        print(f"{backend.name:<20} FINAL: верно {t['correct']}/{cells}, галлюц. {t['hallucination']}, "
              f"пропусков {t['omission']}, неверно {t['wrong']} | валидация ok {stats['ok']}/{n} "
              f"(ретраев {stats['retried']}, провалов {stats['failed']}) | "
              f"RAW до аудита: галлюц. {rt['hallucination']}, пропусков {rt['omission']}, неверно {rt['wrong']}")
        return 0 if t["hallucination"] == 0 else 1
    print(f"== real.py eval | backend={backend.name} | документов: {n} ==\n")
    print(f"{'поле':<15}{'верно':>7}{'галлюц.':>9}{'пропуск':>9}{'неверно':>9}{'точность':>10}")
    for f in SCORED:
        c = per[f]
        print(f"{f:<15}{c['correct']:>7}{c['hallucination']:>9}{c['omission']:>9}{c['wrong']:>9}{c['correct'] / n:>9.0%}")
    print(f"{'warnings (коды)':<15}{warn_ok:>7}{'':>9}{'':>9}{n - warn_ok:>9}{warn_ok / n:>9.0%}")
    t = totals(per)
    cells = n * len(SCORED)
    print(f"\nИТОГО по {cells} полям: верно {t['correct']} ({t['correct'] / cells:.0%})")
    print(f"  ГАЛЛЮЦИНАЦИИ (gold=null, а заполнено): {t['hallucination']}")
    print(f"  ПРОПУСКИ     (gold=значение, а null):  {t['omission']}")
    print(f"  неверное значение (оба не null):       {t['wrong']}")

    ok_c, bad_c = conf_calibration(finals, golds)
    print(f"\nСредняя field_confidence: у верных полей {ok_c}; у неверных {bad_c}")
    print(f"\nВалидация: прошли {stats['ok']}/{n}, из них с ретраем {stats['retried']}, "
          f"провал после {a.max_attempts} попыток {stats['failed']}; всего вызовов модели {stats['attempts_total']}")
    guard = {k[6:]: v for k, v in stats.items() if k.startswith("guard:")}
    print(f"Вмешательства аудита: {dict(guard) if guard else 'нет'}")

    # RAW (то, что модель выдала сама с 1-й попытки) vs FINAL
    parsed = {k: v for k, v in raws.items() if isinstance(v, dict)}
    if parsed:
        rper, _, _ = score({k: parsed[k] for k in parsed}, {k: golds[k] for k in parsed})
        rt = totals(rper)
        print(f"\nRAW (1-я попытка модели, {len(parsed)} из {n} разобрались как JSON, без валидации и аудита):")
        print(f"  галлюцинации {rt['hallucination']}, пропуски {rt['omission']}, неверно {rt['wrong']}")
    print(f"FINAL (после валидации + ретраев + аудита, все {n}):")
    print(f"  галлюцинации {t['hallucination']}, пропуски {t['omission']}, неверно {t['wrong']}")

    if a.verbose or diffs:
        print("\nРасхождения с эталоном (FINAL):" if diffs else "")
        for name, ds in diffs.items():
            for f, c, pv, gv in ds:
                print(f"  {name:<30}{f:<15}{c:<14}pred={pv!r}  gold={gv!r}")
    return 0 if t["hallucination"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
