"""Проверки текста документа для real.py. Только стандартная библиотека.

Что здесь есть:
* normalize_ocr       — исправление OCR-ошибок в числах (O->0, l/I->1);
* find_dates          — даты (12.11.2026, 2026-11-12, «09 қазан 2026 ж.») и их роль;
* find_totals/pick_total — итоги и правило разрешения конфликтов;
* find_bins           — БИН только после метки и только поставщика;
* number_values       — все числа, которые реально написаны в документе (в т.ч. «2 100»).
"""
from __future__ import annotations

import re
from datetime import date

# ------------------------------------------------------------------ OCR

_CONF = {"O": "0", "o": "0", "О": "0", "о": "0", "l": "1", "I": "1", "|": "1"}  # латиница и кириллица
_TOKEN_OK = re.compile(r"^[0-9OoОоlI|][0-9OoОоlI|.,/\-]*$")


def normalize_ocr(raw: str) -> tuple[str, int]:
    """-> (исправленный текст, сколько токенов исправлено).

    Меняем только токены, где есть хотя бы одна настоящая цифра и нет «чужих» букв:
    `I7O34OO2I554` -> `170340021554`, но слово `М4ОО` и форма собственности `ООО` не трогаем."""
    s = raw.replace("\r\n", "\n")
    for ch in "   ":
        s = s.replace(ch, " ")
    state = {"prev_num": False, "fixed": 0}

    def fix(m: re.Match) -> str:
        tok = m.group(0)
        core = tok.rstrip(".,;:)")
        tail = tok[len(core):]
        new = core
        if _TOKEN_OK.match(core) and any(c.isdigit() for c in core) and any(c in _CONF for c in core):
            new = "".join(_CONF.get(c, c) for c in core)
        elif state["prev_num"] and re.fullmatch(r"[Oo]{3}", core):  # «48 OOO» -> «48 000»
            new = "000"
        if new != core:
            state["fixed"] += 1
        state["prev_num"] = bool(re.match(r"^\d[\d.,]*$", new))
        return new + tail

    return re.sub(r"\S+", fix, s), state["fixed"]


def canon(s: str) -> str:
    s = s.casefold()
    for q in "«»\"“”„'`":
        s = s.replace(q, "")
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------ числа

def _f(s: str) -> float:
    return float(s.replace(" ", "").replace(",", "."))


def number_values(text: str) -> set[float]:
    """Все числа, написанные в тексте. «2 100 105 000» даёт и 2, и 2100, и 105000 и т.д.
    (группы по 3 цифры через пробел), поэтому цену «2 100» можно найти в тексте."""
    vals: set[float] = set()
    for line in text.split("\n"):
        toks = [t.rstrip(".,;:") for t in line.split()]
        for i, t in enumerate(toks):
            if not re.fullmatch(r"\d+(?:[.,]\d+)?", t):
                continue
            vals.add(_f(t))
            if re.fullmatch(r"\d{1,3}", t):
                acc = t
                for u in toks[i + 1:]:
                    if not re.fullmatch(r"\d{3}(?:[.,]\d+)?", u):
                        break
                    acc += u
                    vals.add(_f(acc))
                    if re.search(r"[.,]", u):
                        break
    return vals


# ------------------------------------------------------------------ даты

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4, "мамыр": 5, "маусым": 6, "шілде": 7, "тамыз": 8,
    "қыркүйек": 9, "қазан": 10, "қараша": 11, "желтоқсан": 12,
}
_D_NUM = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
_D_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_D_WORD = re.compile(r"(?<!\d)(\d{1,2})\s+([^\W\d_]+)\s+(\d{4})")
# «действительно до 30.11.2026», «оплатить до ...» — это срок, а не дата документа
_EXCLUDED = re.compile(r"(?:действител\w*|срок\w*|оплат\w*|поставк\w*|отгрузк\w*|гаранти\w*)\s*(?:до|по)?\s*:?\s*$|\bдо\s*:?\s*$|дейін\s*$", re.I)
# «от 12.11.2026», «Дата: ...», «Составлено ...»
_PRIMARY = re.compile(r"(?:\bот|дата(?:\s+выставлени\w+)?|составлен\w*|выставлен\w*|күні|dated?)\s*:?\s*$", re.I)


def find_dates(text: str) -> list[tuple[date, str]]:
    """-> [(дата, роль)] в порядке появления. Роль: primary | excluded | other."""
    hits: list[tuple[int, date]] = []

    def add(y, m, d, pos):
        try:
            hits.append((pos, date(int(y), int(m), int(d))))
        except ValueError:
            pass

    for m in _D_NUM.finditer(text):
        add(m[3], m[2], m[1], m.start())
    for m in _D_ISO.finditer(text):
        add(m[1], m[2], m[3], m.start())
    for m in _D_WORD.finditer(text):
        if m[2].lower() in _MONTHS:
            add(m[3], _MONTHS[m[2].lower()], m[1], m.start())
    out = []
    for pos, d in sorted(hits):
        line_start = text.rfind("\n", 0, pos) + 1
        prefix = text[max(line_start, pos - 40):pos]
        kind = "excluded" if _EXCLUDED.search(prefix) else "primary" if _PRIMARY.search(prefix) else "other"
        out.append((d, kind))
    return out


def date_conflict(text: str) -> tuple[date | None, list[date]]:
    """ПРАВИЛО: датой документа считается ПЕРВАЯ «первичная» дата. Если первичных дат несколько и
    они разные — берём первую, а все различные значения возвращаем как конфликт."""
    prim = [d for d, k in find_dates(text) if k == "primary"]
    if not prim:
        return None, []
    distinct = list(dict.fromkeys(prim))
    return prim[0], (distinct if len(distinct) > 1 else [])


# ------------------------------------------------------------------ итоги

_P3 = re.compile(r"всего\s+к\s+оплате|итого\s+к\s+оплате|к\s+оплате|итого\s+с\s+ндс|всего\s+с\s+ндс|төлеуге\s+жиыны|барлығы", re.I)
_P2 = re.compile(r"итого|всего|жалпы\s+сомасы|жиыны|\btotal\b", re.I)
_NUM = re.compile(r"\D*?(\d{1,3}(?: \d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)")


def find_totals(text: str) -> list[tuple[int, float]]:
    """-> [(приоритет, сумма)] по строкам с метками итога. Приоритет 3 — «к оплате»/«итого с НДС»,
    2 — «итого»/«всего». Строки про НДС («Итого НДС», «в т.ч. НДС») итогом не считаются."""
    out = []
    for line in text.split("\n"):
        m3 = _P3.search(line)
        m = m3 or _P2.search(line)
        if not m:
            continue
        if (re.search(r"ндс|ққс|vat", line, re.I) and not m3) or re.search(r"без\s+ндс|в\s*т\.?\s*ч", line, re.I):
            continue
        n = _NUM.match(line, m.end())
        if n and _f(n.group(1)) > 0:
            out.append((3 if m3 else 2, _f(n.group(1))))
    return out


def total_conflict(text: str, vat: float | None) -> tuple[float | None, list[float]]:
    """ПРАВИЛО: побеждает метка высшего приоритета («к оплате» > «итого»); при равенстве — первая по тексту.
    Конфликт — несколько РАЗНЫХ сумм, разницу которых не объясняет НДС
    («итого 160 500 + НДС 25 680 = к оплате 186 180» — это норма)."""
    c = find_totals(text)
    if not c:
        return None, []
    best_prio = max(p for p, _ in c)
    best = next(v for p, v in c if p == best_prio)
    vals = sorted({round(v, 2) for _, v in c})
    if len(vals) > 1 and not (len(vals) == 2 and vat is not None and abs(vals[1] - vals[0] - vat) < 0.01):
        return best, vals
    return best, []


# ------------------------------------------------------------------ БИН

_BUYER = re.compile(r"заказчик|покупател|плательщик|получател|тапсырыс\s*беруші|сатып\s*алушы|customer|buyer|bill\s*to|клиент", re.I)
_BIN = re.compile(r"(?<![^\W\d_])(?:БСН\s*/\s*БИН|БИН\s*/\s*ИИН|БИН|БСН|BIN)\s*[:№#]?\s*(\d(?:[ \-]?\d){11})(?!\d)", re.I)


def find_bins(text: str) -> list[tuple[str, bool]]:
    """-> [(12 цифр, это_БИН_заказчика)]. ИИН физлиц и любые 12 цифр без метки БИН/БСН сюда не попадают."""
    out = []
    for m in _BIN.finditer(text):
        ls = text.rfind("\n", 0, m.start()) + 1
        buyer = bool(_BUYER.search(text[ls:m.start()]))
        if not buyer and ls > 0:  # «Заказчик:» на предыдущей короткой строке
            ps = text.rfind("\n", 0, ls - 1) + 1
            prev = text[ps:ls - 1].strip()
            buyer = len(prev) < 40 and bool(_BUYER.search(prev))
        out.append((re.sub(r"\D", "", m.group(1)), buyer))
    return out


# ------------------------------------------------------------------ прочее

CURRENCY = {
    "KZT": re.compile(r"тенге|теңге|(?<![^\W\d_])тг(?![^\W\d_])|₸|KZT", re.I),
    "USD": re.compile(r"USD|\$|доллар", re.I),
    "EUR": re.compile(r"EUR|€|евро", re.I),
}
_TYPE_WORDS = {
    "invoice": re.compile(r"сч[её]т|шот|invoice", re.I),
    "commercial_offer": re.compile(r"коммерческ\w+\s+предложени\w*|коммерциялық\s+ұсыныс|commercial\s+offer", re.I),
    "act": re.compile(r"(?<![^\W\d_])акт(?:\s|і|$)|актісі", re.I),
}


def doc_type_in_text(text: str, doc_type: str) -> bool:
    return doc_type == "unknown" or bool(_TYPE_WORDS[doc_type].search(text))


def vat_exempt(text: str) -> bool:
    return bool(re.search(r"не\s+облагается|без\s+ндс|ққс[\s\-]*сыз|освобожд", text, re.I))


def warning_code(w: str) -> str:
    return w.split(":", 1)[0].strip()
