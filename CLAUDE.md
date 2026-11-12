# Project Brief: AI Shopping Agents vs. Human Customers (Customer Analytics, IE University)

You are helping me (Mo) build the agent side of a group project for my Customer Analytics class at IE University. The architecture is agreed (see sections 6-8). I want this to end up as a genuinely cool project, both for the class and as a public GitHub portfolio piece.

## 1. The course assignment

- Group project, team of 4. I'm the first one starting, and I own the agent side and the architecture. The others will build on what I set up.
- Deliverables:
  1. **Video presentation (25% of final grade), max 7 minutes.** Business and marketing oriented, pitched to a Head of Marketing who doesn't know analytics. Focus on actionable insights and business value, not methods. Must say what we found, why it matters, and how it changes marketing (targeting, segmentation, recommendations, budget). It must be recorded by us, and AI-generated presentations are not allowed.
  2. **Technical report (20%).** Preferably the dataset plus a Google Colab / Jupyter notebook with code, comments and interpretation. The course gives us a template notebook with sections 1-11. Section 8 (implementation roadmap: per-segment initiatives, campaign details, budget, ROI, success metrics, timeline) is flagged as the most important section.
- Rubric highlights: correctness of analysis 30%, problem framing and depth 25%, analysis and insights 25%. For the video: clarity of insights 30%, business relevance 25%, structure 20%.
- Course documents are in `Instructions /` (note the trailing space in the folder name) and `Class Slides/`. Papers are in `papers/`.

## 2. The idea in one paragraph

A growing share of purchases will be made by AI agents acting for customers. Agents choose differently from humans: they read structured data, they react to platform signals in their own way, and they have no brand loyalty. Our project is an **acquisition story** built on the Olist Brazilian e-commerce dataset. Olist customers almost never come back (3.0% of delivered customers buy twice, verified), so the business wins nearly every sale from scratch. We show (a) which human customer segments bring value and what wins a human's purchase, using real Olist data, and (b) what wins an AI agent's purchase, using a controlled experiment on real Olist products. Then we turn both into a marketing plan. Working hypothesis (to be tested, not assumed): **for an agent, every purchase is a new acquisition.** If agents turn out to be sticky, that is a finding too.

## 3. What the team decided

- **Our professor (Mike) approved the direction.** His guidance: we must still do the customer side properly, but in the submission and especially the video we can put more emphasis on the agent part. He suggested framing it as: today roughly 95-98% of buyers are humans and 2-5% are agents, and the agent share will grow. **This split is an assumption**: label it as such in the report, or find a source for it.
- **Dataset: Olist Brazilian E-Commerce** (Kaggle: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Raw CSVs live in `data/raw/olist/` and are git-ignored (the README links to Kaggle). Verified facts:
  - 99,441 orders, Sep 2016 - Oct 2018, 9 CSVs (customers, orders, order_items, payments, reviews, products, sellers, geolocation, category translation).
  - Products have category plus metadata (name *length*, description length, photo count, dimensions), but **no product names or brands**. 610 products have no category.
  - Reviews are **per order, not per product** (3.3% of orders contain more than one product, 551 orders have duplicate reviews). 41% of reviews have a text comment.
  - Late delivery is the big driver of bad reviews: mean score 2.57 when late vs. 4.21 on time; 7.8% of orders are late.
  - Olist categories are not substitute sets (e.g. `health_beauty` mixes razors and scissors). Product type has to be derived from review text.
- **Focus: acquisition, not churn.** The template lists churn prediction only as an example ("e.g."), so we replace it. The near-zero repeat rate is our hook, not a problem.
- **Currency:** Olist is in R$ at 2016-2018 prices. The experiment shows R$, headroom is reported in % (currency-free), and Section 8 uses one stated R$→€ conversion rate.
- **Research questions (draft):**
  - RQ1: Which customer segments bring the most value?
  - RQ2: What drives a human's product choice and satisfaction (price, rating, delivery time, freight)?
  - RQ3: What drives an AI agent's choice on the same products, how does it differ across models, and how does it differ from humans?
- **Human side (mostly the rest of the team, not my focus right now):**
  - Segmentation: k-means or RFM-style on spend, recency, review behavior, region, category and payment type. Frequency is almost always 1, so lean on monetary value plus behavior.
  - Human choice drivers, replacing churn: a demand model per category using **within-product price variation over time** (product fixed effects, e.g. monthly sales vs. price), as in the course's price elasticity session. A cross-sectional log-log across products is confounded by quality.
  - **Do not use review count as a driver of human sales.** Nearly every delivered order produces a review, so review count ≈ sales and the regression is circular.
  - Optionally a logit for "will this order get a bad review", driven by the gap between promised and actual delivery. Bad reviews hurt the next customer's acquisition, so this stays on-story.
  - Caveat: the human models are observational and correlational. The agent experiment is randomized and causal. We say this openly and treat the comparison as directional.
- **Agent side (my part):**
  - A controlled ACES-style headless experiment where LLM agents repeatedly pick one product out of **8 real Olist products** from each of **2 categories: wristwatches (`watches_gifts`) and printer ink cartridges (`computers_accessories`)**. Product type is identified from review text. Attributes come from the data (price, average review score, review count, promised delivery days, freight) and are then randomized around those real values.
  - Listings carry **neutral labels** ("Watch A" ... "Watch H") because Olist has no product names. The agent can't bring brand priors, so we measure how agents read *attributes*, not brand preference. Product fixed effects should be near zero (a sanity check). Noted as a limitation.
  - Delivery shown to the agent is the **promised** delivery (estimated delivery date minus purchase date), because that is what a buyer sees. Freight is shown separately from price, with "Free shipping" when it is 0.
  - The cartridge prompt states that all listed cartridges are compatible with the user's printer.
- **The central comparisons:**
  1. **Model heterogeneity is a main output.** With neutral labels this becomes: different products win under different models, for different reasons (position, rating, delivery weights). That is our volatility story.
  2. Human vs. agent on the same 8 products: Olist human sales share vs. agent selection share at base attributes. Frame this as "what a structured-data agent does with the same assortment", not as a difference in preferences. Humans saw photos, titles and brands; the agent sees only numbers. The ratings are Olist's post-delivery survey, which probably isn't what shoppers saw on the marketplace listing.
  3. Agent freight vs. price weight (does an agent treat R$1 of freight like R$1 of price?) next to the human literature on shipping-cost aversion.
- **Sponsored:** keep the Sponsored tag in the agent experiment. **Make no human-side Sponsored claim**, neither from Olist (no data) nor from Salvi et al.
- **How it maps onto the template notebook:** 2 problem and RQs / 4 EDA (repeat rate as the hook, review distribution, delivery delays vs. reviews) / 5.1 segmentation / 5.2 human choice drivers / 5.3 agent experiment / 6 findings with the human vs. agent comparison / 7 recommendations / 8 a strategy per human segment (8.2), plus a cross-segment "Agent-ready catalog" initiative (8.3: structured product data, review management, delivery-time accuracy, fewer sponsored placements in agent-facing feeds) with a scenario table at 2% / 5% / 15% agent share, and ROI fed by agent price headroom (8.6) / 9 limitations (simulated agents, neutral labels, observational human data, Brazil 2016-2018).
- **Draft video arc (not final, agent-heavy as Mike prefers):** 30s hook ("almost all your customers buy once, and soon some of them won't be human") / 1.5 min human segments / 1 min what wins a human / 2 min what wins an agent plus the side-by-side comparison / 2 min actions, budget and the agent scenario.

## 4. Literature (read and cite these; PDFs in `papers/`)

1. **ACES: "What Is Your AI Agent Buying? Evaluation, Biases, Model Dependence, & Emerging Implications for Agentic E-Commerce"** (Allouah, Besbes, Figueroa, Kanoria, Kumar). arXiv 2508.02630, `papers/ACES_2508.02630.pdf`. Our main design blueprint. Key findings:
   - Agents concentrate demand on a few "modal" products, and which product wins changes sharply across models and model versions. A model update acts like an external demand shock.
   - Position matters a lot, but the shape is model-specific. In the headless JSON setup the effects are even stronger (e.g. Claude Sonnet 4: Overall Pick takes a 10% baseline to 58%; position 1 vs. 8 is about 7x).
   - A "Sponsored" tag lowers selection probability (at a fixed position), while "Overall Pick" raises it a lot. Scarcity tags are weak or zero.
   - Price, rating and review sensitivity point the way a rational shopper's would, but magnitudes differ a lot by model. Price elasticities of about -1.6 to -2.8.
   - "Price headroom": how much more a seller could charge, without losing share, for a better position, an Overall Pick tag, +0.1 rating, or double the reviews (their Table 11).
   - A seller-side LLM that rewrote product descriptions gained significant market share in about a third of category-model pairs.
2. **Magentic Marketplace** (Microsoft Research). arXiv 2510.25779. Severe "first-proposal bias" (10-30x advantage for speed over quality). Framing only; the codebase is overkill.
3. **Horton, Filippas, Manning, "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?"** arXiv 2301.07543. Justification for treating LLMs as experimental subjects.
4. **Reusens et al., "Would a Large Language Model Pay Extra for a View?"** arXiv 2602.09802. Multinomial logit on LLM choices to get WTP, compared with human benchmarks. Past-choice history and personas shift WTP.
5. **Gui & Toubia, "The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective."** arXiv 2312.15524. Changing a price in the prompt can implicitly change other things the model assumes. Their fix is "unblinding": tell the model the design (e.g. prices are set by the store and vary).
6. **"When Can You Trust Your Synthetic Users?"** arXiv 2609.13148 (not in the folder). LLM conjoint WTP often lands at 50-90% of human values for familiar attributes.
7. **Salvi et al., "Commercial Persuasion in AI-Mediated Conversations."** arXiv 2604.04263. Context only for the video (LLM persuasion of humans toward sponsored products). Not used as a human Sponsored benchmark.

## 5. The ACES design (our starting point)

- **Headless variant (ACES Section 7.2), which is what we use:** the agent gets a ranked JSON array of 8 products and returns a JSON choice plus a short reason.
- **Default headless prompt (ACES Figure 10):**
  > You are a personal shopping assistant helping someone find a good {product category}. They haven't specified particular requirements, so use your best judgment about what would work well for a typical person, and select one product to purchase. Here are the products as a JSON array: {json} Respond ONLY with a JSON object in this exact format: {"selected product number": <1-8>, "product name": "<exact product title>", "reason": "<brief explanation>"} Do not include any other text outside the JSON object.
  - We add one unblinding line (Gui & Toubia): prices and delivery terms are set by the store and vary.
- **Randomization per scenario (ACES Table 1), plus our Olist attributes:**
  - Position: random permutation of the 8 products
  - Sponsored tag: on X listings, X ~ Unif{1..4}
  - Overall Pick tag: on one listing without Sponsored
  - Scarcity tag "Only X remaining": on one listing without Sponsored or Overall Pick, X ~ Unif{1..5}
  - Price: p' = p · f, f ~ LogNormal(0, 0.3)
  - Rating: r' = r + α(5 − r), α ~ Unif[−0.8, 0.8], shown to 1 decimal
  - Number of reviews: N' = N · f, f ~ LogNormal(0, 1)
  - **Ours:** promised delivery days d' = d · f, f ~ LogNormal(0, 0.3), whole days; freight × LogNormal(0, 0.3) with P(free shipping) = 0.2
- **Model settings:** cheapest small tier per provider, reasoning off or minimal, temperature 1.0 where the model allows it (log the actual parameters per model).
- **Analysis:** a conditional logit per model and category:
  U_ij = Σ β_pos·Pos + Σ β_tag·Tag + β_price·ln(Price) + β_rating·Rating + β_rev·ln(Reviews) + β_days·Days + β_freight·(Freight or Free) + θ_j (product FE) + ε.
  Price headroom for an attribute change Δx is exp(β_x·Δx / |β_price|) − 1. Bootstrap CIs clustered by scenario. ACES dropped never-chosen products; we report never-chosen products as a finding.

## 6. Experiments (agreed), in priority order

1. **Choice experiment (priority 1).** ACES randomization above, both categories, every model. Plus a **base-attribute arm** with only position randomized, to get the agent share for comparison with Olist's human share.
2. **Cold-start (priority 2), both categories.** A new listing with very few reviews (e.g. 5) against established ones (hundreds), plus graded advantages for the newcomer: price −5/−10/−20/−30%, faster delivery, Overall Pick tag. Output: a dose-response curve per model, i.e. "what does a new seller need to win an agent's first purchase". This is the acquisition question and lies outside the range the logit covers.
3. **Switching (priority 3, bonus, cartridges only).** Run only if time and budget allow. The agent "already bought" cartridge A ("Last order: X, the user was happy with it"). Graded improvements to a competitor (price −5/−10/−20/−30%, rating +0.1/+0.3, faster delivery, Overall Pick on the competitor, Sponsored on the incumbent), with a no-history control and a no-change arm for the incumbency premium. Watches are excluded because nobody re-buys the same watch.

**Not doing:** screenshots or browser agents, Magentic codebase, buyer-vs-seller multi-agent setups, scarcity-tag variants, reasoning-effort sweeps, more categories. Seller-side description rewrite and persona variants only if everything else is done.

## 7. Engineering requirements

- Clean, reproducible, portfolio-quality repo: config-driven experiments (YAML), seeds, pinned model IDs, and every raw request and response logged (JSONL) so analysis can be re-run without calling APIs again.
- Provider-agnostic agent interface, so adding a model is one config line.
- Async calls with retries and rate-limit handling. Resume by scenario ID.
- API keys in `.env`, never committed. `data/raw/` is git-ignored.
- Robust parsing of the JSON answers, tracking invalid or failed responses with a status code rather than silently dropping them.
- The analysis must be runnable in Colab/Jupyter. The repo holds the engine; the notebook reads committed processed results and raw logs (no API keys needed).
- Clear README with the research question, design, how to run, results and charts. Cite ACES properly.
- **Catalog:** `scripts/build_catalog.py` builds both catalogs reproducibly from the Olist CSVs: delivered orders only, product type from a documented review-keyword regex (spot-checked by hand), at least 30 orders per product, collapse same-seller same-price variants, top 8 by orders. Per product: neutral label, base price (median), rating (mean), review count, promised delivery days (median), freight (median), human sales share, Olist product ID. Output: `configs/catalogs/{watch,cartridge}.yaml` and `data/processed/human_shares.csv`.

### Budget (hard rule)
- **Hard cap $10 total across all runs, target under $5.** This is a uni project, so as close to free as possible.
- Use the cheapest small tier from each provider (mini / flash / haiku type), reasoning off or minimal. Check current pricing and model IDs; use free tiers wherever they exist (e.g. Google AI Studio).
- **Before every run, print the estimated cost and wait for Mo's OK. Never start a run that would push cumulative spend over the cap.** Keep a running spend ledger in the repo.
- Scale scenarios down if needed, but keep enough for the logit to have power.

### Models (final, after the pilot on 2026-11-12)
- `claude-haiku-5-5` (paid, thinking off), `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` (free tier, thinking minimal). The two Gemini lite versions give the "model update as demand shock" comparison.
- Dropped: `gemini-3.6-flash` (free tier allows only ~20 calls/day). Only the cheapest model tier is allowed from any provider.
- Scale: choice experiment Haiku 500 per category, Gemini 300 per category (the same first 300 scenarios); base-attribute arm 100 per category; cold-start 9 arms x 30 per category; cold-start follow-up `coldstart2` (newcomer with 25/50/100 reviews, newcomer at -50% price) 4 arms x 30 per category.
- **Decision 2026-11-12: Haiku is the main subject; Gemini only for choice + base** (the model-comparison slide: position effects, shares). Gemini is not run on cold-start; Haiku answers the acquisition question.
- Gemini switched from the free tier to the paid tier on 2026-11-12 (billing enabled, rpm 60) to finish choice + base the same day. Earlier Gemini calls were free.
- Runs costing under $1 may start without waiting for Mo; above $1, wait for an OK. Commit locally after each run; never push or create a GitHub repo unless Mo asks.

### Analysis plan (after the runs)
- Conditional logit per model and category, headroom with bootstrap CIs (section 5).
- **Stated vs. revealed:** code each reason text by which attributes it mentions (rating, price, reviews, delivery, freight/free shipping, Overall Pick, Sponsored, position) and compare mention rates with the logit effects per model. Position is a likely "revealed but never stated" driver.
- **Charts:** human vs. agent share for the same products; price headroom per attribute per model; position heatmap; cold-start curves; stated vs. revealed.

## 8. Build order and status

1. `scripts/build_catalog.py`, then show Mo both catalogs before anything else.
2. The engine (schemas, scenario generator, providers, runner, parser, cost estimate).
3. A 20-scenario pilot per model (done 2026-11-12). **Stop after the pilot** and report invalid-response rates, actual cost, and a first look at the choices before any full run.
4. Full runs in priority order: choice experiment, cold-start, then switching only if time and budget allow. **Done 2026-11-12** (choice, base, coldstart, coldstart2; total spend $0.68). Switching not run.
5. Next: analysis on the real data and the five charts (headroom in % for cartridges; for watches the price coefficient is too small, so use choice-probability lift), then README and the Colab notebook.
