"""Data schemas: catalog products, scenarios (what the agent sees), raw calls and parsed choices."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]


class Product(BaseModel):
    product_id: str
    label: str
    base_price: float
    base_rating: float
    base_reviews: int
    base_promised_days: int
    base_freight: float
    human: dict = Field(default_factory=dict)
    olist_product_id: str = ""


class Catalog(BaseModel):
    category: str
    display_name: str
    prompt_note: str | None = None
    currency: str = "BRL"
    products: list[Product]

    @classmethod
    def load(cls, key: str) -> "Catalog":
        with open(ROOT / "configs/catalogs" / f"{key}.yaml") as f:
            return cls.model_validate(yaml.safe_load(f))


class Listing(BaseModel):
    """One product as shown to the agent in one scenario (values after randomization)."""

    position: int  # 1..8, order in the JSON array
    product_id: str
    label: str
    price: float
    rating: float  # displayed value (1 decimal)
    reviews: int
    promised_days: int
    freight: float  # 0.0 = free shipping
    sponsored: bool = False
    overall_pick: bool = False
    scarcity: int | None = None  # "Only X left in stock"


class Scenario(BaseModel):
    scenario_id: str
    experiment: str  # choice | base | coldstart | switching
    category: str
    seed: int
    arm: dict = Field(default_factory=dict)  # e.g. {"lever": "price", "dose": -0.1, "focal": "watch_c"}
    history: str | None = None  # switching: purchase-history line added to the prompt
    listings: list[Listing]


class RawCall(BaseModel):
    """Everything sent and received, so analysis never needs to call an API again."""

    call_id: str
    scenario_id: str
    model: str  # key in models.yaml
    model_id: str  # provider model ID requested
    model_version: str | None = None  # version string returned by the provider
    provider: str
    params: dict
    prompt: str
    prompt_hash: str
    response_text: str | None = None
    finish_reason: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_thinking: int | None = None
    cost_usd: float = 0.0
    cost_usd_paid_equivalent: float = 0.0
    latency_ms: int | None = None
    attempts: int = 1
    error: str | None = None
    ts: str


class Choice(BaseModel):
    call_id: str
    scenario_id: str
    model: str
    status: str  # ok | ok_name_mismatch | repaired | invalid | api_error
    chosen_position: int | None = None
    chosen_product_id: str | None = None
    named_label: str | None = None
    reason: str | None = None
