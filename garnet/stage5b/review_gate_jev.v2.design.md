# Stage 5b traced-path review gate — Jev (TypeSafe System One) design, v2

Design for auditing traced lines and branches with **Jev** (`jev-1.13.0`), grounded on a
**fresh measurement** of the current residual. Supersedes the v1 doc
(`review_gate_jev.design.md`), whose labelled set is now invalid — see §2.

---

## 1. The decision the gate makes

Every walk gets one of three actions. Nothing is deleted by a model.

| band | condition | action |
| --- | --- | --- |
| `good` | `P(on_a_drawn_line) >= 0.60` **and** `confidence >= 0.45` | annotate as on-line |
| `unsure` | `0.35 <= P < 0.60`, or `confidence < 0.45` | keep, mark `needs_human` |
| `bad` | `P(on_a_drawn_line) < 0.35` | keep, mark `likely_bad` |

**Jev annotates; it never deletes.** A wrong reject silently removes topology; a wrong
accept is visible and recoverable. Destructive authority stays in the deterministic layer
(R1's `reject`, R2's geometry) plus the human queue.

---

## 2. Why v1's ground truth could not be reused

v1 was measured on **14 residual walks/sheet** over `0002A`, `0003`, `0007`, with four
hand labels (`002A branch_000013` BAD, `0007 branch_000007` BAD, `0007 obj_000084` GOOD,
`0007 equip_d_2506:port_02` GOOD).

Between then and now the tracer changed materially (dash-vs-solid snap, perpendicular leg
finding, jump bridging). Re-measuring the same 14 against the current output:

```
2 unchanged, 6 changed, 6 no longer in the residual at all
```

`0007 branch_000007` moved **coverage 0.58 → 0.98, unexplained 61px → 2px** — it was the
v1 BAD label and is now clean. `0007 equip_d_2506:port_02` moved 1px → 21px. Two of the four
labels are gone and two moved.

**So the v1 labels are void.** The residual is now **53 walks across 9 sheets (~6/sheet)**,
not 14/sheet on three sheets. All numbers below are from the current run.

---

## 3. What Jev can and cannot see

Jev is **text only** — no images. So the pixel evidence must be serialised. The crop is
never sent; the *measurements* in it are.

**The state is defined by construction: an "unexplained" run has no ink under it.** Measured
across all 101 runs in the 53 residual walks, **101/101 carry zero ink within 2px** inside
the run. That is what "unexplained" means — the definition, not a finding.

So the model is never asked *whether* ink is missing; it is asked **what the missing stretch
is**, which is exactly what a pixel test cannot answer. Two earlier attempts to classify the
runs from pixels were both wrong, and both are recorded here because they are the reason the
model is needed:

- A **dash-train test** (`n_ink_runs >= 3`) reported "43% dash trains" — but any neighbouring
  text supplies 3 runs. Refuted.
- A **duty-cycle test** over a ±30px window reported "61% dash trains" — but the window
  included the walk's **own adjacent segments**, so the duty was measuring the walker's
  earlier travel, not the gap. Refuted: `0010 branch_000010` scored duty 0.40 "MIXED" while
  sitting at x=1452 with **0 ink even at 2px for 90 consecutive px**.

Both fail the same way: a local pixel window cannot tell blank paper from a symbol's interior
from a text label, because in all three cases the walk itself covers nothing. That is the
judgement Jev is asked for.

**Everything countable stays in code** — coverage, pixel totals, run lengths, nearest
symbols and their distances. Jev is asked only the judgement.

---

## 4. The state sent (one per walk, ~570–620 input tokens)

```json
{
  "walk": {"id": "...", "kind": "branch", "sheet": "...",
           "segment_count": 3, "trace_length_px": 724, "terminal_type": "branch_connection"},
  "measured_on_the_drawing": {
    "fraction_of_path_with_ink_within_2px": 0.80,
    "fraction_of_path_centred_on_ink": 0.33,
    "fraction_of_path_over_3px_from_any_ink": 0.29,
    "unexplained_blank_total_px": 45,
    "longest_single_unexplained_px": 12,
    "stretches_with_no_drawn_line_and_no_known_symbol": [
      {"length_px": 12, "from_xy": [..], "to_xy": [..],
       "known_symbols_within_60px": [{"class": "gate valve", "distance_px": 9.0}]}
    ]
  },
  "note": "An unexplained stretch is ..."
}
```

Filtered deliberately: only the 4 longest stretches, only symbols within 60px, top 3 each.
v1 sent a longer state; the skill's own guidance is that accuracy falls as irrelevant
state grows.

---

## 5. The questions (two, not three)

### `line_fidelity` — Choice over the ordered spectrum

The **primary** judgement. Written directly and in positive polarity, with the
interruption stated **in the instructions** rather than left as a state field.

```
instructions: "This traced path was produced by a pipe-tracing algorithm on a P&ID. Its
  unexplained stretches are listed under stretches_with_no_drawn_line_and_no_known_symbol,
  with the pixel lengths and the known symbols near them. Weigh those stretches first.
  Which describes how this traced path relates to the drawn pipe linework?"

criteria:
  on_a_drawn_line                        the path runs along a drawn pipe line the whole
                                         way; any unexplained stretches are short or are
                                         inside a symbol the detector missed
  mostly_on_the_line_with_a_long_break   the path runs along a drawn pipe line but includes
                                         at least one long stretch where no drawn line is present
  beside_the_linework                    the path runs parallel to or beside drawn linework
                                         rather than on it
  over_blank_paper_or_text               the path runs over blank paper, or across printed
                                         characters and labels, rather than along a drawn line
```

v1's failure was a Noul asking a **negated** question: 0.17–0.44 across good and bad alike.
Two doc-listed failure modes compounded — contradictory instruction-vs-criteria, and
indirection. v2 asks directly, as a Choice, because `on_a_drawn_line` is a *position*, and
Choice is relative (settles *which*) where a Noul is absolute.

### `interruption_cause` — Choice

Asks what explains the **longest** stretch, as a separate judgement. This is the field that
carries the actual discriminative detail (text vs undetected component vs jump vs blank).

### Deliberately not asked

- **`terminal` validity** — R3 already answers it by set lookup. Asking a model to do a
  dictionary lookup adds a failure mode for zero information.
- Anything countable — Jev is not a calculator.
- Rationale — Jev produces none. A generative model writes prose from Jev's verdict.

---

## 6. Scoring: atomic answers, composed in code

```python
# the only place thresholds and weights live
W_SPECTRUM, W_OCCUPANT = 0.80, 0.20

def walk_score(answers):
    p = answers["line_fidelity"]["probabilities"]
    good = p.get("on_a_drawn_line", 0.0)          # the driver
    cause = answers["interruption_cause"]["probabilities"]
    benign = (cause.get("an_undetected_component_or_symbol", 0.0)
              + cause.get("a_drawn_line_jump_or_break", 0.0))
    score = W_SPECTRUM * good + W_OCCUPANT * benign
    # Band on the CALIBRATED PROBABILITY, never the unvalidated weighted composite.
    band = "good" if good >= 0.60 else ("bad" if good < 0.35 else "unsure")
    if answers["line_fidelity"]["confidence"] < 0.45:
        band = "unsure"
    return score, band
```

---

## 7. Verification — measured, not assumed

### 7.1 Stability: **1 flip in 53**

Two identical runs over the same 53 states:

```
band flips: 1/53   (0007A equip_v_2502:port_07, bad -> unsure)
|delta P(on_a_drawn_line)|: mean 0.015, max 0.110
```

Stable enough to threshold. **Not zero** — report it as 1/53, never "deterministic". This is
exactly why the verdict set is non-destructive.

### 7.2 Redundancy: **Jev is NOT a repackaged threshold**

The necessary test before crediting a model with anything. Rank correlation of each measured
feature against `P(on_a_drawn_line)`:

| feature | Spearman rho |
| --- | --- |
| `worst_run_px` | **−0.849** (best) |
| `unexplained_total_px` | −0.798 |
| `coverage` | +0.572 |
| `n_runs` | −0.413 |

A single threshold agrees with Jev on only **77%** of walks at the best cut, and the
disagreements are systematic, not noise. The clearest case:

```
0004A equip_d_2511:port_05   coverage 0.80   unexplained 45px   P(line)=0.17  -> bad
```

`unexplained_px` ranks it *good* (45px spread over 7 short runs, worst only 12px). The crop
shows why Jev is right: the walk runs **through the printed label `2" BLANKOFF (UTILITY)`**,
and every character it crosses counts as ink within 2px, so the metric scores it as covered.
No length threshold can see this. Jev can, because the state tells it the stretch has text
nearby and none of the interruptions are symbols.

Symmetrically `0010 obj_000124` (unexplained 10px, P=0.85 good) is ranked bad by the
threshold. So the model is adding information, not restating the metric.

### 7.3 Cost

```
53 walks, 2 questions batched per call:  59,236 input tokens = $0.00249 for all 9 sheets
```

Input is the only billed side ($0.042/Mtok, output free).

---

## 8. Open items

1. **The v1 label set is void and the new set is only partly labelled.** Hand-verified so
   far: `0004A equip_d_2511:port_05` BAD, `0002A branch_000013` BAD, `0004 branch_000033`
   BAD, `0009 branch_000025` GOOD, `0003 equip_d_2502:port_06` GOOD. **The required
   asymmetry — no hand-labelled BAD walk in `good` — has not yet been asserted over a real
   sample.** Label ~20 walks before trusting the bands.
2. **Weights `0.80/0.20` and cuts `0.60/0.35/0.45` are provisional** and inherit nothing
   from v1's `0.70/0.15/0.15` — the skill is explicit that a threshold must not be carried
   between primitive shapes or across a data change.
3. **Per-sheet batching** would cut cost further (14×2 question ids against one state
   object per sheet). Not needed at $0.0003/sheet.
4. **`confidence < 0.45 → unsure` fired on several `good`-band walks.** Check whether that
   cut is costing good walks a human glance unnecessarily, or genuinely catching the
   ambiguous middle.
5. **Pin `jev-1.13.0`, never the alias.** Log `model` from every response — it is the only
   way to attribute a later drift.

---

## 9. The honest scope

Of 80 walks on a sheet, ~6 reach Jev; the rest are settled by R1/R2/R3 in code. Jev's job is
a narrow learned tiebreaker on the ambiguous middle, not a model-driven judge. Sending all
80 would add a failure mode for no information.
