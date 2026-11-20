from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

import checks

MODEL = "openai/gpt-oss-120b"
MAX_ATTEMPTS = 3
SEMANTIC_CODES = {"CONFLICT_DATE", "CONFLICT_TOTAL", "OCR_FIXED", "VAT_EXEMPT"}  # дедуп по коду


# ====================================================================== схемы

class ReceiptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Название товара или услуги")
    qty: Optional[float] = Field(default=None, description="Количество")
    price: Optional[float] = Field(default=None, description="Цена за 1 штуку")


class LLMDetectOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_type: Literal["invoice", "commercial_offer", "act", "unknown"] = Field(
        description="Тип текста: счет на оплату, торговое предложение, акт или неизвестный документ"
    )
    supplier_name: Optional[str] = Field(default=None, description="наименование организации/поставщика/компании")
    supplier_bin: Optional[str] = Field(default=None, description="БИН-бизнес-идентификационный номер организации")
    doc_number: Optional[str] = Field(default=None, description="номер документа")
    doc_date: Optional[str] = Field(default=None, description="Дата в формате YYYY-MM-DD")
    currency: Optional[Literal["KZT", "EUR", "USD"]] = None
    total_amount: Optional[float] = Field(default=None, description="Итоговая сумма к оплате")
    vat_amount: Optional[float] = Field(default=None, description="Итоговая сумма налога")
    line_items: list[ReceiptItem] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list, description='Противоречия в документе: "CONFLICT_DATE: ..." / "CONFLICT_TOTAL: ..."'
    )

    @field_validator("doc_date")
    @classmethod
    def _iso_date(cls, v: Optional[str]) -> Optional[str]:
        # Неверный формат/несуществующая дата -> ValidationError -> ретрай с текстом ошибки
        if v is not None:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
                raise ValueError(f"doc_date должна быть в формате YYYY-MM-DD, получено {v!r}")
            date.fromisoformat(v)
        return v


class FinalReceiptOutput(LLMDetectOutput):
    field_confidence: dict[str, float] = Field(default_factory=dict)


# ====================================================================== LLM

SYSTEM_PROMPT = """Ты извлекаешь поля из текста делового документа (счёт, коммерческое предложение, акт).
Верни ТОЛЬКО один JSON-объект строго по схеме, без пояснений и без markdown.
ЖЁСТКИЕ ПРАВИЛА
1. Чего нет в документе — null (для line_items — пустой список). Лучше null, чем догадка.
2. Не выводи БИН из названия компании. БИН — только 12 цифр после метки БИН/БСН/BIN и только
у ПОСТАВЩИКА/ИСПОЛНИТЕЛЯ. БИН заказчика/покупателя, ИИН физлиц и любые другие 12 цифр — не подходят.
3. Не подставляй сегодняшнюю дату. doc_date — дата самого документа («от ...», «Дата», «Составлено»).
Сроки («действительно до», «оплатить до») — не дата документа. Неполная дата («октябрь 2026») — null.
4. НДС (vat_amount) — только если сумма НДС написана в документе. Не считай её по ставке.
«НДС не облагается» -> vat_amount = null.
5. total_amount — итоговая сумма к оплате с НДС: метка «к оплате»/«итого с НДС» важнее «итого».
Если итог есть только прописью — переведи прописью в число. Если итога нет — null (не суммируй позиции).
6. Противоречия (две разные даты документа, два разных итога) не скрывай: добавь в warnings строку
вида "CONFLICT_DATE: ..." / "CONFLICT_TOTAL: ...". Значение — первая по тексту дата;
для итога — по правилу из п.5, при равенстве — первая по тексту.
7. Ошибки OCR (O/0, l/I/1) в числах и БИН исправляй.
8. doc_type: invoice | commercial_offer | act | unknown. Накладная, письмо и всё прочее — unknown.
9. currency: KZT (тенге/тг/₸), USD, EUR или null.
10. Двуязычное название поставщика (каз./рус.) — бери русскую форму.
11. supplier_name — название организации, которая выставила документ (поставщик/исполнитель/продавец),
вместе с организационно-правовой формой, как в документе: «ТОО "Название"», «АО «Название»». Организация из подписи
письма или из текста, не относящегося к счёту/КП/акту/накладной, поставщиком не является — тогда null.
12. doc_number — только сам номер, без слов «№», «No», «Ne».
"""


def build_messages(raw_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<document>\n{raw_text}\n</document>"},
    ]


_client = None


def groq_call(messages: list[dict]) -> str:
    """Один вызов Groq. Импорт ленивый, чтобы отсутствие пакета или ключа давало понятный BACKEND_ERROR."""
    global _client
    if _client is None:
        from groq import Groq

        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("не задана переменная окружения GROQ_API_KEY")
        _client = Groq(api_key=key, timeout=60)
    response = _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "receipt_extraction", "schema": LLMDetectOutput.model_json_schema()},
        },
        temperature=0.0,
    )
    return response.choices[0].message.content


def extract_json(raw: str) -> str:
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("в ответе нет JSON-объекта")
    try:
        json.loads(raw[a : b + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"невалидный JSON: {e.msg} (строка {e.lineno}, столбец {e.colno})") from e
    return raw[a : b + 1]


# ====================================================================== аудит

def audit_and_enrich(llm_data: LLMDetectOutput, raw_text: str) -> tuple[FinalReceiptOutput, list[str]]:
    """Python проверяет то, что сказала модель. -> (результат, список вмешательств для статистики)."""
    text, n_ocr = checks.normalize_ocr(raw_text)   # проверяем по тексту с исправленным OCR
    d = llm_data.model_dump()
    warnings: list[str] = list(d.pop("warnings"))  # то, что написала сама модель
    fixes: list[str] = []
    confidence: dict[str, float] = {}

    def drop(name: str, code: str, why: str) -> None:
        d[name] = None
        warnings.append(f"{code}: {why}")
        fixes.append(f"null:{name}")

    # ---- тип документа
    if d["doc_type"] != "unknown" and not checks.doc_type_in_text(text, d["doc_type"]):
        d["doc_type"] = "unknown"
        warnings.append("UNGROUNDED_DOC_TYPE: в тексте нет слов, подтверждающих тип документа")
        fixes.append("downgrade:doc_type")
    if d["doc_type"] == "unknown":
        confidence["doc_type"] = 0.50
        warnings.append("Документ не распознан")
    else:
        confidence["doc_type"] = 0.95

    # ---- номер: убрать «№»/«No»/«Ne», которые модель иногда оставляет; название и номер должны быть в тексте
    if d["doc_number"]:
        d["doc_number"] = re.sub(r"^(?:№\s*|N[oOеe°]\.?\s+)", "", d["doc_number"].strip()) or None
    if d["supplier_name"] and checks.canon(d["supplier_name"]) not in checks.canon(text):
        drop("supplier_name", "UNGROUNDED_NAME", "названия поставщика нет в тексте")
    if d["doc_number"] and d["doc_number"].lower() not in text.lower():
        drop("doc_number", "UNGROUNDED_NUMBER", "номера документа нет в тексте")
    confidence["supplier_name"] = 0.95 if d["supplier_name"] else 0.0
    confidence["doc_number"] = 0.95 if d["doc_number"] else 0.0
    if not d["supplier_name"]:
        warnings.append("Не удалось определить название Компании")

    # ---- БИН: 12 цифр, есть в тексте, принадлежит поставщику
    bin_ = d["supplier_bin"]
    if bin_ is None:
        confidence["supplier_bin"] = 0.0           # в КП БИНа может просто не быть — это не ошибка
    elif not re.fullmatch(r"\d{12}", bin_):
        drop("supplier_bin", "INVALID_BIN", "БИН не состоит из 12 цифр")
        confidence["supplier_bin"] = 0.0
    else:
        found = [buyer for v, buyer in checks.find_bins(text) if v == bin_]
        if not found:
            drop("supplier_bin", "UNGROUNDED_BIN", "такого БИН нет в тексте рядом с меткой БИН/БСН")
            confidence["supplier_bin"] = 0.0
        elif all(found):
            drop("supplier_bin", "UNGROUNDED_BIN", "этот БИН принадлежит заказчику/покупателю, а не поставщику")
            confidence["supplier_bin"] = 0.0
        else:
            confidence["supplier_bin"] = 0.99

    # ---- дата: есть в тексте, это не срок; противоречия
    if d["doc_date"]:
        roles = [k for dt, k in checks.find_dates(text) if dt.isoformat() == d["doc_date"]]
        if not roles:
            drop("doc_date", "UNGROUNDED_DATE", "такой даты нет в тексте")
        elif all(k == "excluded" for k in roles):
            drop("doc_date", "UNGROUNDED_DATE", "это срок («действительно до»), а не дата документа")
    first_date, date_conf = checks.date_conflict(text)
    if date_conf:
        warnings.append(
            "CONFLICT_DATE: несколько разных дат документа ("
            + ", ".join(x.isoformat() for x in date_conf) + "); взята первая по тексту"
        )
        if d["doc_date"] != first_date.isoformat():
            d["doc_date"] = first_date.isoformat()
            fixes.append("override:doc_date")
    confidence["doc_date"] = 0.95 if d["doc_date"] else 0.0

    # ---- суммы: должны быть написаны в документе (НДС не считаем, итог не суммируем)
    nums = checks.number_values(text)
    for name, code in (("total_amount", "UNGROUNDED_TOTAL"), ("vat_amount", "UNGROUNDED_VAT")):
        v = d[name]
        if v is not None and not any(abs(v - x) < 0.005 for x in nums):
            drop(name, code, f"суммы {v:g} нет в тексте (возможно, посчитана моделью)")
    best_total, total_conf = checks.total_conflict(text, d["vat_amount"])
    if total_conf:
        warnings.append(
            "CONFLICT_TOTAL: несколько разных итоговых сумм (" + ", ".join(f"{x:g}" for x in total_conf)
            + "); взята с высшим приоритетом метки, при равенстве — первая"
        )
        if d["total_amount"] is None or abs(d["total_amount"] - best_total) > 0.005:
            d["total_amount"] = best_total
            fixes.append("override:total_amount")
    if d["vat_amount"] is None and checks.vat_exempt(text):
        warnings.append("VAT_EXEMPT: в документе указано, что НДС не облагается; vat_amount = null (не 0, не считаем)")

    # ---- валюта
    if d["currency"] and not checks.CURRENCY[d["currency"]].search(text):
        drop("currency", "UNGROUNDED_CURRENCY", f"маркера валюты {d['currency']} нет в тексте")

    # ---- позиции: название и числа должны быть в тексте
    ctext = checks.canon(text)
    kept = []
    for it in d["line_items"]:
        if checks.canon(it["name"]) not in ctext:
            warnings.append(f"UNGROUNDED_ITEM: позиции {it['name']!r} нет в тексте — удалена")
            fixes.append("drop:line_item")
            continue
        for k in ("qty", "price"):
            if it[k] is not None and not any(abs(it[k] - x) < 0.005 for x in nums):
                warnings.append(f"UNGROUNDED_ITEM: {k}={it[k]:g} позиции {it['name']!r} нет в тексте — обнулено")
                fixes.append(f"null:line_item.{k}")
                it[k] = None
        kept.append(it)
    d["line_items"] = kept

    # ---- арифметика: итог == сумма позиций ИЛИ сумма позиций + НДС
    complete = bool(kept) and all(i["qty"] is not None and i["price"] is not None for i in kept)
    if d["total_amount"] is None:
        confidence["total_amount"] = 0.0
    elif complete:
        subtotal = sum(i["qty"] * i["price"] for i in kept)
        vat = d["vat_amount"]
        ok = abs(d["total_amount"] - subtotal) <= 0.05 or (vat is not None and abs(d["total_amount"] - subtotal - vat) <= 0.05)
        confidence["total_amount"] = 0.99 if ok else 0.45
        if not ok:
            warnings.append(f"Расхождение: по позициям {subtotal:g}, в итоге {d['total_amount']:g}")
    else:
        confidence["total_amount"] = 0.75   # позиций нет/неполные: арифметикой не проверить

    confidence["vat_amount"] = 0.95 if d["vat_amount"] is not None else 0.0
    confidence["line_items"] = 0.9 if kept else 0.0
    if d["currency"] is not None:
        confidence["currency"] = 0.95
    else:
        confidence["currency"] = 0.0

    # ---- OCR: числа в тексте были исправлены -> доверие к числовым полям ниже
    if n_ocr:
        warnings.append(f"OCR_FIXED: в тексте исправлены вероятные OCR-ошибки в числах ({n_ocr} мест)")
        for k in ("supplier_bin", "doc_date", "total_amount", "vat_amount", "doc_number"):
            confidence[k] = min(confidence[k], 0.7)

    # ---- дедуп предупреждений (по коду для смысловых, иначе по тексту)
    seen, uniq = set(), []
    for w in warnings:
        key = checks.warning_code(w) if checks.warning_code(w) in SEMANTIC_CODES else w
        if key not in seen:
            seen.add(key)
            uniq.append(w)

    return FinalReceiptOutput(**d, field_confidence=confidence, warnings=uniq), fixes


# ====================================================================== пайплайн

@dataclass
class RunInfo:
    ok: bool                       # False => вернули пустой результат (ответ модели не прошёл валидацию / сбой API)
    attempts: int
    raw_first: Optional[str]       # сырой ответ 1-й попытки (для eval: что модель выдала сама)
    errors: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)


def empty_result(warning: str) -> FinalReceiptOutput:
    """Безопасный результат: ничего не выдумываем."""
    return FinalReceiptOutput(doc_type="unknown", warnings=[warning])


def run_pipeline(
    raw_text: str, call_llm: Callable[[list[dict]], str] = groq_call, max_attempts: int = MAX_ATTEMPTS
) -> tuple[FinalReceiptOutput, RunInfo]:
    messages = build_messages(raw_text)
    errors: list[str] = []
    raw_first = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = call_llm(messages)
        except Exception as e:  # нет ключа, сеть, лимиты, таймаут: ретрай тут не лечит
            msg = f"{type(e).__name__}: {e}"
            return empty_result(f"BACKEND_ERROR: {msg}"), RunInfo(False, attempt, raw_first, errors + [msg])
        raw_first = raw if raw_first is None else raw_first
        try:
            parsed = LLMDetectOutput.model_validate_json(extract_json(raw))
        except (ValueError, ValidationError) as e:   # ValidationError — подкласс ValueError
            if isinstance(e, ValidationError):
                err = "; ".join(f"{'.'.join(map(str, x['loc'])) or '<root>'}: {x['msg']}" for x in e.errors()[:6])
            else:
                err = str(e)
            errors.append(f"попытка {attempt}: {err}")
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Твой ответ не прошёл программную проверку:\n{err}\n"
                                            "Верни ТОЛЬКО исправленный JSON по той же схеме. Чего нет в документе — null."},
            ]
            continue
        final, fixes = audit_and_enrich(parsed, raw_text)
        return final, RunInfo(True, attempt, raw_first, errors, fixes)
    last = re.sub(r"\s+", " ", errors[-1])[:300]
    return (
        empty_result(f"VALIDATION_FAILED: ответ модели не прошёл валидацию за {max_attempts} попыток ({last})"),
        RunInfo(False, max_attempts, raw_first, errors),
    )


def process_document(raw_text: str) -> str:
    final, _ = run_pipeline(raw_text)
    return final.model_dump_json(indent=2)
