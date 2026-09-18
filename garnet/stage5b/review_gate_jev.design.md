# Stage 5b trace review gate — Jev (TypeSafe System One) design

Design for auditing traced lines and branches with **Jev** (`jev-1.13.0`, TypeSafe System One).
Supersedes the vision-model residual in `review_gate.py`.

## Why this is a redesign, not a swap

The existing gate hands a **base64 image crop** to a vision model for R1's residual. Jev's
hard limit rules that out:

> **Text only** (`string`/`object`/`array`). No image, audio, video. Anything pixel-based is out.

So the pixel evidence must be **serialised into text** before it can be judged. That is the
central design move: `review_gate.py` already computes everything Jev needs as numbers
(`coverage`, `on_line`, `unexplained_px`, gap runs with their positions and what bbox explains
them) — those numbers are the state. Nothing about the crop is sent, because Jev cannot see it.

Two further Jev facts shape the design:

- **No arithmetic, does not count reliably.** Every count, ratio, sum and threshold stays in code.
  Jev only judges what a number cannot settle.
- **Structural invariants do not hold.** A Noul and an equivalent yes/no Choice disagree (docs'
  own example: Noul `0.22` vs Choice `yes=0.01/no=0.99`), and a question plus its negation do not
  sum to 1 (`0.72` + `0.47`). So: never carry a threshold between primitives, never ask the same
  question twice in two shapes, and never assume two answers compose arithmetically.

## What stays deterministic (the majority)

Measured over **102 walks** on the three runnable sheets (`0002A`, `0003`, `0007`):

| rule | decided | result |
| --- | --- | --- |
| R1 (solid line) | 88 of 102 | 86 `accept`, 2 `reject` — 14 to Jev |
| R2 (one line per place) | **102 of 102** | pure geometry, Jev never consulted |
| R3 (valid terminal) | **102 of 102** | pure lookup, Jev never consulted |

R2 and R3 are already fully deterministic and should stay that way. R3's 19 rejects are dead
ends (`dead_end` 15, and similar) — set membership, not judgment. Sending those to Jev would be
asking a model to do a dictionary lookup.

**So Jev sees 14 of 102 walks (13.7%).** That is the honest scope: the gate is a deterministic
filter with a narrow learned tiebreaker, not a model-driven judge.

## What the 14 residual walks actually look like

They are exactly the ambiguous middle: mostly-on-ink paths whose interruptions are small enough
that a local pixel test cannot say whether the interruption is an inline component or a printed
label.

```
002A branch_000013  cov 0.45 unexp  35px   <- real pipe through a reducer, or a symbol-side seed?
002A branch_000016  cov 0.82 unexp  47px
002A branch_000027  cov 0.70 unexp  19px
002A branch_000033  cov 0.74 unexp  24px
002A equip_p_2504a:port_02 cov 0.97 unexp 26px
002A equip_v_2501:port_02  cov 0.75 unexp 38px
0003 equip_d_2502:port_03  cov 0.88 unexp 32px
0007 branch_000001  cov 0.93 unexp  65px
0007 branch_000006  cov 0.92 unexp  71px
0007 branch_000007  cov 0.58 unexp  61px
0007 branch_000012  cov 0.74 unexp  15px
0007 obj_000084     cov 1.00 unexp   5px
0007 obj_000088     cov 0.97 unexp   9px
0007 equip_d_2506:port_02 cov 1.00 unexp 1px
```

Note the spread: `equip_d_2506:port_02` has **1px** unexplained and is probably fine;
`branch_000006` has **71px** and is probably not. Gap size alone does not decide it — that is
exactly why these 14 and not the other 88.

## Jev question set — v1 FAILED, v2 tested

**v1 was a Noul asking a negated question. It did not separate anything.** Measured on all 14
residual walks, `noul` sat in **0.17–0.44 for good and bad walks alike**:

```
002A branch_000013   cov=0.45 unexp=35px  noul=0.31   <- pixel-verified BAD
0007 branch_000007   cov=0.58 unexp=61px  noul=0.44   <- pixel-verified BAD
0007 obj_000084      cov=1.00 unexp= 5px  noul=0.21   <- probably GOOD
0007 branch_000006   cov=0.92 unexp=71px  noul=0.24   <- 71px over paper, still "on a line"
```

Two doc-listed failure modes compounded:

- **#7 contradictory instruction vs criteria** — the instruction asserted a *negation*
  ("...the interruptions are text ... so the path is **not** following a drawn line"). The
  docs' own example is that a Noul whose `true` means "no" performs worse.
- **#4 indirection / double negatives** — the question asked the model to reason through
  "interruptions are not components therefore not on a line".

The docs say to write directly. **v2 asks the question directly, as a Choice over the ordered
spectrum** — which is also the correct primitive, because `on_a_drawn_line` is a *position*, and
Choice is **relative** (settles *which*) while a Noul is absolute.

### v2 — Choice over the spectrum (this is the one to build)

```
instructions: "Which describes how this traced path relates to the drawn pipe linework?"
criteria:
  on_a_drawn_line
      "the path runs along a drawn pipe line the whole way, allowing for inline components
       it continues through"
  on_a_drawn_line_with_a_long_break
      "the path runs along a drawn pipe line but includes a long stretch where no drawn
       line is present"
  beside_the_linework
      "the path runs mostly parallel to or beside drawn linework rather than on it"
  over_blank_paper_or_text
      "the path runs over blank paper or across printed characters"
```

Isolated check on a synthetic good/bad pair:

| case | choice | confidence | P |
| --- | --- | --- | --- |
| GOOD (cov 1.00, 1px blank) | `on_a_drawn_line` | 0.92 | 0.95 correct |
| BAD (cov 0.58, 61px blank) | `on_a_drawn_line_with_a_long_break` | 0.75 | 0.81 correct |

That is real separation, on the *right* option each time.

### Measured on the 14 real residual walks

```
sheet walk                    cov  unexp  P(line) conf  choice                                band
002A  equip_p_2504a:port_02  0.97    26     0.64 0.51  on_a_drawn_line                       good
002A  equip_v_2501:port_02   0.75    38     0.34 0.50  ..._with_a_long_break                 bad
002A  branch_000013          0.45    35     0.21 0.64  ..._with_a_long_break                 bad   hand=BAD  PASS
002A  branch_000016          0.82    47     0.26 0.65  ..._with_a_long_break                 bad
002A  branch_000027          0.70    19     0.10 0.81  ..._with_a_long_break                 bad
002A  branch_000033          0.74    24     0.33 0.55  ..._with_a_long_break                 bad
0003  equip_d_2502:port_03    0.88    32     0.43 0.40  ..._with_a_long_break                 unsure
0007  obj_000084             1.00     5     0.64 0.52  on_a_drawn_line                       good  hand=GOOD PASS
0007  obj_000088             0.97     9     0.57 0.43  on_a_drawn_line                       unsure
0007  equip_d_2506:port_02    1.00     1     0.67 0.57  on_a_drawn_line                       good  hand=GOOD PASS
0007  branch_000001          0.93    65     0.06 0.90  ..._with_a_long_break                 bad
0007  branch_000006          0.92    71     0.61 0.48  on_a_drawn_line                       good  <-- WRONG
0007  branch_000007          0.58    61     0.03 0.66  ..._with_a_long_break                 bad   hand=BAD  PASS
0007  branch_000012          0.74    15     0.67 0.56  on_a_drawn_line                       good
```

**Required asymmetry holds: no hand-labelled BAD walk reaches `good` (4/4 correct).**

### The one known wrong answer, and its cause

`0007 branch_000006` has **71px** of unexplained blank — more than either verified-bad walk — and
Jev called it `on_a_drawn_line` (P=0.61). Its state lists a 71px interruption with
`inside_known_symbols: []` and the note *"no known symbol covers this stretch"*, so the evidence
was present and the verdict went the wrong way.

Note the confound: `branch_000006`'s **coverage is 0.92** while the verified-bad `branch_000007`
is 0.58. The model appears to weigh coverage heavily and the interruption note weakly. That is
the thing to fix — not by reweighting alone, but by making the interruption the *primary*
evidence in the instruction rather than a field the model must notice.

### Stability — 1 flip in 14, not zero

Two identical runs over the same states:

```
choice flips: 1/14
P(on_a_drawn_line) mean |delta|: 0.035   max: 0.080
```

Stable enough to threshold, but **not** perfectly. Report it as 1/14, not "deterministic". This
is why the model keeps a non-destructive verdict set.



### Deliberately NOT asked

- Anything countable (walk length, gap count, ratio) — Jev does not count reliably; code does it.
- Anything a set lookup answers (is `dead_end` a valid terminal) — code does it.
- Several judgments in one question — the signal to decompose, not to combine.
- Rationale — Jev produces no explanations. If a human needs a reason, a generative model writes
  it from Jev's verdict; do not ask Jev for prose.

## Scoring system: atomic answers → one auditable score

Jev returns atomic judgments; **code composes them**. Weights live in one reviewable file next
to the questions (docs' own guidance), never inside a prompt.

```python
# the only place thresholds and weights live
W_SPECTRUM, W_OCCUPANT, W_TERMINAL = 0.70, 0.15, 0.15

def walk_score(answers) -> tuple[float, str]:
    """Composite 0..1, higher = more trustworthy as real pipe."""
    probs = answers["spectrum"]["probabilities"]
    good = probs.get("on_a_drawn_line", 0.0)          # spectrum dominates
    component = answers["occupant"]["probabilities"].get("component_symbol", 0.0)
    noul = answers["terminal"]["noul"]
    terminal_ok = max(noul, 1 - noul)                 # noul has NO confidence field
    score = (W_SPECTRUM * good
             + W_OCCUPANT * component
             + W_TERMINAL * terminal_ok)
    # Band on the spectrum probability, not the composite: banding on a weighted
    # sum mixes an uncalibrated weight into a calibrated probability.
    band = ("good" if good >= 0.60 else "bad" if good < 0.35 else "unsure")
    if answers["spectrum"]["confidence"] < 0.45:
        band = "unsure"                               # low option-ranking certainty
    return score, band
```

The spectrum weight dominates deliberately: it is the only component that separates the
hand-labelled walks. `occupant` and `terminal` are corroboration, not drivers — and both were
measured to add nothing on this set.

### The confidence gate is separate from the score

Doc guidance: answer says *what*, confidence says *whether to act*. Thresholds scale with risk.
**Gate on the spectrum's probabilities and `confidence`, never on the composite alone.**

| band | condition | action |
| --- | --- | --- |
| `good` | `P(on_a_drawn_line) >= 0.60` and spectrum `confidence >= 0.45` | annotate as on-line |
| `unsure` | `0.35 <= P < 0.60`, or `confidence < 0.45` | keep, mark `needs_human` |
| `bad` | `P(on_a_drawn_line) < 0.35` | keep, mark `likely_bad` |

**Jev annotates; it never deletes.** The docs and the earlier gate agree from opposite
directions: a wrong `reject` silently removes topology, a wrong `accept` is visible and
recoverable, and the model is not deterministic (measured: 1/14 flips). So the model's verdict
set here is **non-destructive**. Destructive authority stays in the deterministic layer (R1's
`reject`, R2's geometry) plus the human queue.

### Why a composite AND a band on one component

The composite is for ranking and for a single audit number. The band is for the decision,
because it is read off the *calibrated probability* rather than a weighted sum whose weights are
unvalidated. Conflating them would let an uncalibrated weight decide a threshold — the docs warn
against treating levels/weights as numerically meaningful when they are not.


## Audit record

One row per walk, in `review_jev_audit.json`. Every field is either measured or answered — no
field is inferred.

```json
{
  "walk_id": "branch_000013",
  "sheet": "14780-8120-25-25-0002A",
  "kind": "branch",
  "deterministic": {"r1": "review", "r2": "accept", "r3": "accept",
                    "coverage": 0.45, "unexplained_px": 35, "longest_run_px": 33},
  "jev": {
    "model": "jev-1.13.0",
    "answers": {
      "on_drawn_line":  {"noul": 0.31, "certainty": 0.69},
      "occupant_choice": {"choice": "component_symbol", "probabilities": {...},
                          "confidence": 0.58},
      "line_fidelity":  {"score": 1.42, "probabilities": {...}, "confidence": 0.41}
    },
    "usage": {"input_tokens": 612, "output_tokens": 44}
  },
  "score": 0.41,
  "band": "medium",
  "final": "flag",
  "reason": "on-line certainty 0.69 (<0.65 gate); composite 0.41 in medium band"
}
```

Requirements the audit record must satisfy:

- **Log the versioned model id** (`jev-1.13.0`), never the alias — an alias moves under you and
  previously-tuned thresholds silently drift. Pin the version in code.
- **Record `usage.input_tokens`** — input is the only thing billed ($0.042/Mtok; output free).
- **Record which rules were decided in code** so a reviewer can see Jev touched 14 of 102.
- **Record both the score and its components**, so a wrong verdict can be attributed to a
  weighting rather than to the model.

## Verification before trusting it

The existing gate's discipline applies unchanged, and is what makes any threshold meaningful:

1. **Stability first.** Two identical runs must give the same verdicts. (Local measurement on
   `jev-1.13.0`: 0/39 verdict flips, noul moved a mean of 0.011 — the stability that makes a
   gate possible. A vision model tested here flipped verdicts on unchanged input.)
2. **Label a set by hand, then measure.** The old gate used 12 walks (6 bad, 6 good). 14
   residual walks on three sheets is not enough to tune three weights; label more before
   trusting the bands.
3. **Assert the asymmetry that matters:** no hand-labelled BAD walk may reach `accept`. A GOOD
   walk landing in `flag` only costs a human glance; a BAD walk accepted is topology corruption.
4. **Check the wording didn't leak the answer.** Paraphrasing a threshold from the code into the
   prompt made a model agree 100% — a tautology, not a measurement. If agreement on the labelled
   set climbs toward 100%, suspect the prompt contains the conclusion.
5. **Do not carry a threshold between primitives.** A cut tuned on the Noul is not valid on the
   equivalent Choice.

## Cost

Measured: ~300–620 input tokens per walk for this state size, output free.

- 14 residual walks/sheet ≈ **9k input tokens ≈ $0.0004/sheet**.
- Even sending **all 102** walks (the wasteful thing we are explicitly not doing) ≈ 50k tokens
  ≈ **$0.002** — the same order as the docs' own 39–50-decision batches.

Cost is not the constraint; the constraint is that 88 of 102 walks have a deterministic answer
and sending them to a model would add a failure mode for no information.

## Open items — for Jev review

1. **`branch_000006` is a known wrong answer** (71px unexplained blank called `on_a_drawn_line`).
   The fix is to make the interruption the *primary* evidence in the instruction rather than a
   state field the model may under-weight, then re-measure. Do not paper over it with a weight.
2. **Weights `0.70/0.15/0.15` and bands `0.60/0.35/0.45` are provisional.** They are consistent
   with 4 hand labels; they are not calibrated. Label more walks (the old gate used 12) before
   trusting them, and expect the `occupant` and `terminal` components to be dropped if they keep
   adding nothing.
3. **Stability is 1/14, not 0.** Re-measure after any wording change, and keep every threshold
   gated behind a human queue until it is 0 over a larger set.
4. **Per-sheet batching.** All 14 walks of a sheet could share one call with 14×3 question ids,
   since questions are evaluated independently and share state. Needs each walk's state to be
   addressable within one `state` object; would cut cost further.
5. **Whether R1's two deterministic `reject`s should route to the human queue** rather than being
   dropped outright — a deterministic rejection of real pipe is still silent topology loss.
6. **Pin `jev-1.13.0`, never `jev-latest`.** An alias moves under you and silently drifts the
   thresholds. Log the versioned id on every response.

## Cost — measured

```
14 residual walks/sheet, batched 3 questions per call:  14,065 input tokens total  =  $0.00059
```

Input is the only billed side ($0.042/Mtok; output free). Sending **all 102** walks — the wasteful
thing this design explicitly avoids — would be ~50k tokens ≈ $0.002, the same order as the docs'
own 39–50-decision batches.

Cost is not the constraint. The constraint is that **88 of 102 walks have a deterministic answer**,
and sending them to a model would add a failure mode for zero information.

