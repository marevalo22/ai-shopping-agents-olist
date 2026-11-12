"""Parse agent responses into choices. Nothing is dropped silently: every call gets a status.

ok                valid JSON, number in range, name matches the listing at that number
ok_name_mismatch  valid number, but the product name points at a different listing (we keep the number)
repaired          no valid JSON, but a number and/or exact label could be recovered from the text
invalid           nothing usable
api_error         the call itself failed
"""

from __future__ import annotations

import json
import re

from .schemas import Choice, RawCall, Scenario

NUM_KEYS = ("selected product number", "selected_product_number", "product number", "product_number", "selected")
NAME_KEYS = ("product name", "product_name", "title", "name")


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    for m in re.finditer(r"\{.*?\}", text, flags=re.S):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, flags=re.S)  # greedy fallback (nested braces)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _get(obj: dict, keys: tuple[str, ...]):
    low = {k.lower().strip(): v for k, v in obj.items()}
    for k in keys:
        if k in low:
            return low[k]
    return None


def parse(call: RawCall, scenario: Scenario) -> Choice:
    base = {"call_id": call.call_id, "scenario_id": call.scenario_id, "model": call.model}
    if call.error or call.response_text is None:
        return Choice(**base, status="api_error")

    n = len(scenario.listings)
    by_pos = {li.position: li for li in scenario.listings}
    by_label = {li.label.lower(): li for li in scenario.listings}
    text = call.response_text

    obj = _extract_json(text)
    if obj is not None:
        num, name, reason = _get(obj, NUM_KEYS), _get(obj, NAME_KEYS), obj.get("reason")
        try:
            num = int(str(num).strip())
        except (TypeError, ValueError):
            num = None
        named = by_label.get(str(name).strip().lower()) if name else None
        if num is not None and 1 <= num <= n:
            li = by_pos[num]
            status = "ok" if named is None or named.product_id == li.product_id else "ok_name_mismatch"
            if named is None and name:
                status = "ok_name_mismatch"
            return Choice(**base, status=status, chosen_position=num, chosen_product_id=li.product_id,
                          named_label=str(name) if name else None, reason=reason)
        if named is not None:
            return Choice(**base, status="repaired", chosen_position=named.position,
                          chosen_product_id=named.product_id, named_label=named.label, reason=reason)

    # Free-text fallback: exactly one listing label mentioned
    hits = {li.product_id: li for li in scenario.listings if li.label.lower() in text.lower()}
    if len(hits) == 1:
        li = next(iter(hits.values()))
        return Choice(**base, status="repaired", chosen_position=li.position, chosen_product_id=li.product_id,
                      named_label=li.label)
    return Choice(**base, status="invalid")
