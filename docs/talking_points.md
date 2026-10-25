# Talking points

Eight judgement calls worth raising in an interview. Each follows the same shape: what I
noticed, the evidence, the decision, and what it cost. Numbers are HI-Small test at the
0.5% alert budget unless stated; the experiment evidence is in
[experiments.md](experiments.md).

---

## 1. Cutting the simulator's artificial tail

**Noticed.** Test's laundering rate was 0.177%, against 0.075% in train. Counting per day
showed that after Sep 10 volume falls from hundreds of thousands of payments a day to a few hundred,
and **59% of those are laundering**: the simulator stops legitimate activity while
unfinished schemes keep playing out.

**Decision.** Drop every row at or after 2022-09-11 in ingestion, before any id exists.
Otherwise the test set rewards spotting "the network went quiet", which any activity
count encodes for free. Test's rate became 0.113%, in line with validation's 0.107%.

**Cost.** 655 of 5,177 positives (13%), and the late hops of about a quarter of the
schemes. *How you'd catch it on real data:* plot volumes and label rates over time, and
compare base rates across splits, before modelling.

---

## 2. Point-in-time features, and the tests that prove them

**Noticed.** Leakage in AML features is easy to miss and flattering. A rolling count that
includes the current payment, or others in the same minute, quietly encodes the scheme
being scored: a fan-out hub's 8 payments in one minute would each "see" the other 7.

**Decision.** Windows are `[t − w, t)`, with pandas rolling `closed="left"` on per-account
event timelines. Payments in the same minute are treated as simultaneous, because
minute-resolution data can't order them. Two tests prove it:
- **Future perturbation:** rewrite every row after a cutoff, and earlier features must
  not change by a bit.
- **Same-minute twins:** give every payment in a minute a twin, and that minute's
  originals must not change.

**Why two tests.** I checked the tests by breaking the code on purpose, widening the
window to `closed="both"`. The future-perturbation test still passed: it only compares
rows *before* the cutoff, so it can't see a same-minute leak. The twin test was added
for that, and it fails under the same break.

**Cost.** A burst of 10 payments in one minute is invisible to each of them. That's the
price of correctness at minute resolution.

---

## 3. The rules baseline catches 0.9%, and that's realistic

**Noticed.** Six textbook scenarios with a priori thresholds (structuring, velocity,
fan-in, pass-through, cross-currency, new account + large payment) reach **0.9% recall
and 0.20% precision**: 1 alert in 500 is real, about twice random. Bank rules commonly
run above 95% false positives.

**Evidence per rule.**
- **Velocity** (≥10 payments in 24h) fires on 27% of rows, because AMLworld has many
  busy legitimate accounts.
- **Cross-currency** catches no laundering at all.
- **Calibrated rules:** fitting thresholds and weights on the training split only lifts
  validation recall from 1.1% to 1.8%. So "the model beats mis-set rules" is ruled out.

**Decision.** Keep the rules as the bar, scored with **expected tie-breaking**. The
weighted score has only 7 distinct values, so thousands of rows tie at the cut-off.

**Cost.** None. Rules stay useful as readable reason codes alongside the model.

---

## 4. Class weighting broke the model

**Noticed.** The textbook move for 0.1% positives, up-weighting them √(neg/pos) ≈ 36×,
produced a **one-tree model**, with identical precision at two different budgets.

**Evidence.** Validation AP logged tree by tree crashed after tree 1, even on training
data. With no weighting, AP rose to about 0.26.

**Decision.** No reweighting. Alert-budget metrics care only about the order of the top
few thousand rows, and large leaf values chasing rare positives scramble that order.

**Cost and the general check.** Later, sqrt and balanced weighting, undersampling and
SMOTE were each tuned like everything else. All of them landed inside the no-weighting
interval, and all lowered PR-AUC. A tiny tree count, or identical precision across
budgets, is now treated as a red flag in every run.

---

## 5. Where the gain comes from

**Evidence** (validation recall at 0.5%, 30 Optuna trials per feature set):

| Change | Recall |
|---|---|
| Raw transaction fields | 37% |
| + point-in-time account history and degrees | **70.5%** |
| + windowed counterparties, cycles, amount statistics | 70.6% at most |

The graph groups lift PR-AUC (0.477 → 0.509), but only further down the list than a
0.5% budget reaches.

**Other findings.**
- **Tuning:** on the same 45 features, tuning took test recall from 43% to 58%.
- **Model families:** XGBoost and LightGBM are indistinguishable. Logistic regression
  trails by 30 points, because the signal is in thresholds and interactions.
- **Isolation Forest scores 0.6%, below the rules.** Laundering payments look normal one
  at a time; the pattern across accounts and time needs labels.

**Decision.** Ship the simplest set at the top: the 45 account-history features.

---

## 6. Reason codes showed the model mostly asks "is it ACH?"

**Noticed.** Per-alert SHAP reasons are for investigators, but read in bulk they audit the
model. **"Paid by ACH" is the first reason on 99% of alerts.**

**Evidence.**
- ACH is 14% of payments but 87% of laundering.
- The 13% of laundering paid by cheque, card, cash or Bitcoin gets 47 alerts and **no
  hits**.
- Payment format has the largest mean |SHAP|.

**Decision.** Report it as a known weakness rather than tune it away. It's a property of
the simulator, and a real bank's mix differs, so a model that learned "ACH = suspicious"
would transfer badly. Next step: recall by payment format next to typology recall, and a
separate model for the non-ACH queue.

**Cost.** None to find it, which is the point. Without the explanations I'd have
reported 58% recall and not known this.

---

## 7. Account age drifted by construction, so it was capped

**Noticed.** Daily drift monitoring (PSI per feature, against the worst training day)
flagged account age on every day after training. It means "time since first seen in the
data", so it can't exceed the time since the data began: training never shows the ages
test has.

**Decision.** Cap it at 48h, and make it missing for the data's first 48h, when every
account looks young only because the data just began. 48h rather than 24h, because with a
24h cap 17% of senders still looked new on day 2.

**Measured against an alternative.** "Time since first activity in the last 48h" looked
purer, but it treats accounts back from a quiet weekend as new: 3–23% "new", swinging by
weekday.

**Adoption rule, written before the run.** Adopt unless validation recall falls below the
old model's 95% interval (67.1%). It scored 69.1%. On test, 58.3% against 58.5%, and the
paired interval contains 0.

**Cost.** Nothing measurable, and the model now means the same thing on scoring day as
on training day.

---

## 8. The runner-up beat the winner on test, and I didn't switch

**Noticed.** The selection rule was written before any test score existed: best validation
recall, with ties inside the confidence interval going to the simpler model. It picked
the account-history set (70.5%) over the runner-up with windowed counterparties (70.6%).
On test, the runner-up won by **2.4–4.5 points in every seed**, with paired intervals
excluding zero.

**Decision.** Keep the pre-registered choice, and make the runner-up the candidate for the
next round on fresh data. Choosing after seeing test turns test into a second validation
set, and then no honest number is left.

**Cost.** About 3 points of test recall this time. That's the price of a rule that can't
be bent after the fact. *Why paired intervals:* the two models' separate intervals
overlap, but they score the same rows, so the interval of the difference is far tighter.
