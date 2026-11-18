"""Live price agent: an LLM with web search finds current Malaysian prices (RM) for each kennel action,
returns structured rows with sources, and writes a dated cost table the optimizer reads.

Usage:
    export ANTHROPIC_API_KEY=...        (or OPENAI_API_KEY=... with --provider openai)
    python src/price_agent.py --state "Pulau Pinang"

If the call fails or a price looks implausible, the agent keeps the last cached value for that row,
so results stay reproducible for anyone rerunning the repo without an API key.
"""
import argparse, datetime as dt, json, os, re, sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COSTS = ROOT / "costs"
LATEST = COSTS / "malaysia_costs_latest.csv"

ITEMS = [  # (action, animal, unit, what to search for)
    ("vaccinate", "dog", "per_animal", "core vaccination for one dog at a vet clinic"),
    ("vaccinate", "cat", "per_animal", "core vaccination for one cat at a vet clinic"),
    ("deworm", "dog", "per_animal", "deworming one dog"),
    ("deworm", "cat", "per_animal", "deworming one cat"),
    ("sterilize", "dog_male", "per_animal", "neutering a male dog"),
    ("sterilize", "dog_female", "per_animal", "spaying a female dog"),
    ("sterilize", "cat_male", "per_animal", "neutering a male cat"),
    ("sterilize", "cat_female", "per_animal", "spaying a female cat"),
    ("treat_minor_injury", "any", "per_animal", "a vet consultation plus basic wound treatment and medication"),
    ("pro_photography", "any", "per_session", "a 1 to 2 hour professional pet photography session"),
    ("pro_photography_per_animal", "any", "per_animal", "professional photos of one pet (smallest package)"),
    ("grooming", "any", "per_animal", "basic grooming of one dog or cat"),
    ("daily_care", "dog", "per_day", "daily cost to keep one dog at a shelter (food and care); monthly sponsorship / 30 is acceptable"),
    ("daily_care", "cat", "per_day", "daily cost to keep one cat at a shelter (food and care); monthly sponsorship / 30 is acceptable"),
]
FIELDS = ["action", "animal", "unit", "price_low_rm", "price_mid_rm", "price_high_rm", "source_name", "source_url", "notes"]

PROMPT = """You are pricing services for an animal shelter in {state}, Malaysia. Use web search to find CURRENT prices in
Malaysian ringgit (RM) for each item below. Prefer Malaysian vet clinic price lists, SPCA or NGO pages, and Malaysian news
from the last two years. For each item give a low, typical (mid) and high price, the single best source (name and URL),
and a short note on what the price covers.

Items (keep the action/animal/unit keys exactly):
{items}

Reply with ONLY a JSON array of objects with keys: {fields}. Prices must be numbers in RM, no currency symbols."""


def _items_text():
    return "\n".join(f"- action={a}, animal={b}, unit={u}: {d}" for a, b, u, d in ITEMS)


def ask_anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    for _ in range(5):  # long searches can return stop_reason "pause_turn"; send the turn back to let it continue
        msg = client.messages.create(
            model=os.environ.get("PRICE_AGENT_MODEL", "claude-sonnet-5-5"), max_tokens=8000,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 20,
                    "user_location": {"type": "approximate", "country": "MY"}}],
            messages=messages)
        if msg.stop_reason != "pause_turn":
            break
        messages = [messages[0], {"role": "assistant", "content": msg.content}]
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def ask_openai(prompt):
    from openai import OpenAI
    client = OpenAI()
    resp = client.responses.create(model=os.environ.get("PRICE_AGENT_MODEL", "gpt-5"),
                                   tools=[{"type": "web_search"}], input=prompt)
    return resp.output_text


def parse(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in model reply")
    rows = pd.DataFrame(json.loads(m.group(0)))
    missing = set(FIELDS) - set(rows.columns)
    if missing:
        raise ValueError(f"missing fields {missing}")
    for c in ["price_low_rm", "price_mid_rm", "price_high_rm"]:
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    return rows[FIELDS]


def validate(new, cached, max_ratio=3.0):
    """Keep a new price only if it is positive, ordered low <= mid <= high, and within max_ratio of the cached mid."""
    out, log = [], []
    cached = cached.set_index(["action", "animal"])
    for key in [(a, b) for a, b, _, _ in ITEMS] + [k for k in cached.index if k not in [(a, b) for a, b, _, _ in ITEMS]]:
        old = cached.loc[key] if key in cached.index else None
        cand = new[(new.action == key[0]) & (new.animal == key[1])]
        ok = False
        if len(cand):
            r = cand.iloc[0]
            ok = (r.price_mid_rm > 0 and r.price_low_rm <= r.price_mid_rm <= r.price_high_rm
                  and (old is None or 1 / max_ratio <= r.price_mid_rm / old.price_mid_rm <= max_ratio))
        if ok:
            out.append(r.to_dict()); log.append((key, "updated", r.price_mid_rm))
        elif old is not None:
            out.append({"action": key[0], "animal": key[1], **old.to_dict()}); log.append((key, "kept cached", old.price_mid_rm))
    return pd.DataFrame(out)[FIELDS], log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="Selangor")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    args = ap.parse_args()
    cached = pd.read_csv(LATEST)
    prompt = PROMPT.format(state=args.state, items=_items_text(), fields=", ".join(FIELDS))
    try:
        text = (ask_anthropic if args.provider == "anthropic" else ask_openai)(prompt)
        new = parse(text)
    except Exception as e:
        print(f"Price agent failed ({e}); keeping cached table {LATEST.name}", file=sys.stderr)
        return
    table, log = validate(new, cached)
    stamp = dt.date.today().isoformat()
    table.to_csv(COSTS / f"malaysia_costs_{stamp}.csv", index=False)
    table.to_csv(LATEST, index=False)
    for key, status, price in log:
        print(f"{key[0]:28s} {key[1]:10s} {status:12s} RM{price}")


if __name__ == "__main__":
    main()
