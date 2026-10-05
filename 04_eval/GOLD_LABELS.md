# Gold labels

Ground truth for the eval harness. Fill every field by **human review only**.
Leave any value you are unsure of as `null` — a guessed label is worse than a
missing one, because it silently invalidates the score.

Each field below lists: its type, its valid values, and the question you answer.

- `is_genuine` — boolean (`true` / `false`). Is the signal's central claim actually true?
- `confidence` — number, `0.0` through `1.0`. How confident *should* verification be that this claim is real?
- `stale_presented_as_new` — boolean (`true` / `false`). Does the signal package an already-old event as if it were new?
- `rank` — positive integer; **`1` is the highest priority** (most worth acting on). Where does this signal rank against the other labelled signals? Ranks must be unique and contiguous (1, 2, 3, … with no gaps) across the labelled set.
- `maturity` — integer `1` through `5`. What maturity score should the Evaluation Agent assign?
- `relevance` — integer `1` through `5`. What curriculum-relevance score should the Evaluation Agent assign?
- `action_tier` — string; one of the four tiers the Recommendation Agent actually emits: `watch`, `update_existing_material`, `add_optional_content`, `add_new_lesson`. Which action should the Recommendation Agent take for this signal?

> Note on `action_tier`: `02_src/schemas.py` declares a fifth value,
> `investigate_larger_change`, in `ActionTier`, but `_select_tier` in
> `02_src/agents/recommendation.py` never returns it. It is therefore **not** a
> valid gold value here — labelling it would never match agent output.
