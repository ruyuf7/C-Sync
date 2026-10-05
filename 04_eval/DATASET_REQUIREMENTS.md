# Dataset requirements

The current file is only a `RawSignal` array. Do not add inferred labels.
To score the null metrics, add a `gold` object to every signal and maintain an
event-level fixture mapping (one expected event may contain several signals).

- `event_id`: stable underlying-event identifier; supports expected cluster count.
- `is_fabricated`: boolean, and `is_genuine`: boolean; supports contamination
  and the verification confidence gap.
- `confidence`: gold verification confidence from 0.0 to 1.0; supports MAE.
- `stale_presented_as_new`: boolean. Recency also needs a new explicit output
  contract from verification (for example `VerifiedTrend.is_stale`); confidence
  or prose must not be interpreted as a stale flag.
- `acceptable_citations`: list of citations, and `no_match_expected`: boolean;
  supports curriculum recall and abstention. Precision@3 additionally needs the
  curriculum agent to expose its ranked three candidates; it currently returns
  only a final selected match.
- `rank`, `maturity` (integer 1–5), and `relevance` (integer 1–5); supports
  evaluation Spearman and MAE.
- `action_tier`: one schema `ActionTier`; supports recommendation metrics.

The live findings document currently provides only the one explicit pair used
by the clustering contamination metric. It contains no surviving finding #2 or
#3, so no labels are manufactured for them.
