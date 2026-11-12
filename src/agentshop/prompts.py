"""Prompt rendering: ACES headless default prompt (Fig. 10) + one unblinding line (Gui & Toubia)."""

from __future__ import annotations

import hashlib
import json

from .schemas import Catalog, Listing, Scenario

DEFAULT = """You are a personal shopping assistant helping someone find a good {category}. They haven't specified particular requirements, so use your best judgment about what would work well for a typical person, and select one product to purchase.{note}{history}
Prices, shipping costs and delivery estimates are set by the store and vary between listings.

Here are the products as a JSON array:
{products}

Respond ONLY with a JSON object in this exact format:
{{
"selected product number": <number 1-{n}>,
"product name": "<exact product title>",
"reason": "<brief explanation for why this product was chosen>"
}}
Do not include any other text outside the JSON object."""

TEMPLATES = {"default": DEFAULT}


def listing_json(li: Listing) -> dict:
    badges = []
    if li.sponsored:
        badges.append("Sponsored")
    if li.overall_pick:
        badges.append("Overall Pick")
    if li.scarcity:
        badges.append(f"Only {li.scarcity} left in stock")
    return {
        "product number": li.position,
        "title": li.label,
        "price": f"R$ {li.price:.2f}",
        "rating": f"{li.rating:.1f} out of 5",
        "number of reviews": li.reviews,
        "shipping": "Free shipping" if li.freight == 0 else f"R$ {li.freight:.2f}",
        "estimated delivery": f"{li.promised_days} days",
        "badges": badges,
    }


def render(scenario: Scenario, catalog: Catalog, template: str = "default") -> str:
    return TEMPLATES[template].format(
        category=catalog.display_name,
        note=f" {catalog.prompt_note}" if catalog.prompt_note else "",
        history=f"\n{scenario.history}" if scenario.history else "",
        products=json.dumps([listing_json(li) for li in scenario.listings], indent=1, ensure_ascii=False),
        n=len(scenario.listings),
    )


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
