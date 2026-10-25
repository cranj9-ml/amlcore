# Walkthrough: the project in 10 minutes

A spoken-length tour, in five parts. Each part has what to say, the one number to
remember, and the follow-up question it usually draws. The sections match the slides
one to one.

---

## Part 1. The problem

### 1. Title

**Say:** "This is a transaction-monitoring project on IBM's synthetic AML data. The
question is practical: if investigators can only review half a percent of payments, how
much of the laundering can we put in front of them? I compare the rules most banks run
with a gradient-boosted model, on the same leak-free features, on a later held-out period."

### 2. Why "recall at an alert budget"

**Say:** "Investigators work a queue of fixed size, so a model's job is ranking. I score
the top 0.5%, about 5,000 alerts a day here, and ask what share of all laundering is in
it. At a fixed budget, precision and recall rise together, so there's nothing to trade
off. The real trade-off is the budget itself, which is a staffing decision. I also
report PR-AUC for the whole ranking. Never accuracy: with 0.1% positives, flagging
nothing is 99.9% accurate."

**Remember:** budget 0.5% = 5,078 test alerts. With 1,143 laundering payments in test,
the best possible precision is 22.5%.

**Follow-up:** *"Why not F1?"* F1 needs a threshold, and the real decision is "review the
top k". Recall at k answers that directly. (Best validation F1 is about 0.50, for
comparison with papers.)

---

## Part 2. The data, and two traps

### 3. AMLworld, and the tail I cut

**Say:** "HI-Small has about 5 million payments over 10 days. Plotting volume per day
showed something odd: after Sep 10 traffic collapses to a few hundred payments a day,
and 59% of them are laundering. The simulator stops legitimate activity, but unfinished
schemes keep running. Any activity feature would catch that for free, so I cut
everything from Sep 11 before assigning any ids."

**Remember:** 59% laundering in the tail. Cutting it brought the test base rate from
0.177% down to 0.113%.

**Follow-up:** *"How would you spot that on real data?"* Plot volumes and label rates over
time, and compare base rates across the splits, before modelling.

### 4. Point-in-time features, and the proof

**Say:**
- "Every feature for a payment at time t uses only payments strictly before t: windows
  are [t − w, t). Same-minute payments are treated as simultaneous, because the data
  has minute resolution and can't order them."
- "Each payment becomes two events: an outflow for the sender and an inflow for the
  receiver. Per-account rolling windows with `closed="left"` do the rest."
- "Two tests prove it. One rewrites the future and checks no earlier feature moves. The
  other adds same-minute twins and checks the originals don't move. I checked the tests
  themselves by breaking the code on purpose: the first one alone misses a same-minute
  leak, which is why the second exists."

**Remember:** 45 features. Leakage is tested by mutation, not just asserted.

**Follow-up:** *"What do you lose by excluding the same minute?"* A burst of payments
within one minute is invisible to each of them. That's the cost of correctness at
minute resolution.

---

## Part 3. Baseline against model

### 5. The rules baseline

**Say:**
- "Six textbook scenarios: structuring just under $10k, velocity, fan-in,
  pass-through, cross-currency, and a large payment from a new account. Thresholds are
  set a priori, and a weighted sum ranks the alerts."
- "At 0.5% they catch 0.9% of laundering, and 1 alert in 500 is real. That's realistic:
  bank rules commonly run above 95% false positives."
- "It isn't a straw man. Fitting the thresholds on training data only moves validation
  recall from 1.1% to 1.8%."

**Remember:** rules at 0.9% recall, 0.2% precision.

**Follow-up:** *"Why keep rules?"* Regulators understand them, and each one is a readable
reason. Here they're shown next to the model's reasons in the alert queue.

### 6. The model

**Say:** "LightGBM on the 45 features. It's tuned on validation only, with early
stopping on validation average precision, and has no class reweighting. That last part
was a finding: up-weighting the 0.1% positives gave a one-tree model, because it
scrambles the order at the top of the list, which is the only part a budget looks at."

**Remember:** no reweighting. The weighted model collapsed to 1 tree.

### 7. Results

**Say:** "On the test period: 58% recall at 0.5%, precision 13%, PR-AUC 0.37. That's
about 65 times the rules at the same workload. At account level, which is how
investigators open cases, 40% of laundering accounts are caught and almost 1 in 3
alerted accounts is laundering."

**Show:** `figures/recall_vs_budget.png`

**Remember:** 58.3% [53–63%], 65× the rules, 1 alert in 8 real.

**Follow-up:** *"Validation said 70%, why is test 58%?"* Validation was tuned on and used
for early stopping, so it's optimistic. Test is a later period, scored once, and it's
the number I quote.

### 8. Where the gain comes from

**Say:** "I added feature groups one at a time, under the same tuning budget. Raw
transaction fields give 37%. Adding what each account did before (counts, sums, time
since its last payment, counterparties met so far) gives 70%. Graph patterns on top
(windowed counterparties, cycles, amount statistics) improve the ranking further down
the list, but not the top 0.5%."

**Show:** `figures/feature_ladder.png`

**Remember:** 37% → 70% from account history. Tuning added 15 points on test.

### 9. What didn't help

**Say:** "XGBoost and LightGBM tie. Logistic regression is 30 points behind, because
the signal is in thresholds and interactions. Every imbalance trick, including SMOTE,
undersampling and class weights, landed inside the no-weighting interval with lower
PR-AUC. And unsupervised Isolation Forest scored below the rules: laundering payments
look normal one at a time."

**Follow-up:** *"Can't you just do anomaly detection?"* That's measured here: 0.6%.

---

## Part 4. Trust: can the alerts be explained and relied on?

### 10. Reason codes, and what they revealed

**Say:** "Each alert gets its top three reasons from TreeSHAP, which LightGBM computes
exactly. They're written as sentences with the typical value, like 'receiver made 42
payments in the past 7 days (typical: 3)'. Read in bulk, they audit the model: 'paid by
ACH' is the first reason on 99% of alerts. ACH is 14% of payments but 87% of
laundering in this simulator, and the laundering paid any other way is never caught.
Without the explanations I wouldn't have known."

**Show:** `figures/shap_importance.png`, and three rows of `alerts_test.csv`.

**Follow-up:** *"What would you do about it?"* Report recall by payment format, and try a
separate model for the non-ACH queue. A real bank's mix differs, so this matters for
transfer.

### 11. Typologies, and the weak spot

**Say:** "Documented schemes are caught at 69 to 90%. The weak spot is 'unattributed'
laundering, which belongs to no documented scheme: 40% of test positives, and only 18%
caught. I also flag that schemes run for 1 to 6 days and cross the split cuts, so part
of the typology recall is recognising a scheme you saw start."

**Show:** `figures/typology_recall.png`

**Remember:** documented 69–90%, unattributed 18%.

### 12. Drift, and the account-age fix

**Say:**
- "Daily drift monitoring flagged account age on every day after training. It's 'time
  since first seen in the data', so it grows with the calendar: training never sees
  test's ages."
- "I capped it at 48 hours and made it missing for the data's first 48 hours. I measured
  an alternative first: a rolling 'first active in the last 48h' swung with the
  weekday."
- "I adopted the fix by a rule written before the run. Test is unchanged, 58.3% against
  58.5%, and now the feature means the same thing on every day."

**Remember:** fixed at no cost, and adopted by a pre-written rule.

---

## Part 5. Judgement, and what's next

### 13. The runner-up beat the winner, and I didn't switch

**Say:** "My selection rule, written before any test score, picked the simpler feature
set because the richer one's validation lead was inside its interval. On test the
richer one won by about 3 points in every seed. I kept the pre-registered choice and
made the runner-up the candidate for the next round. Switching after seeing test turns
test into validation, and then I have no honest number left."

**Remember:** about 3 points was the price of discipline. Paired intervals made the
difference visible.

### 14. Limitations, and what I'd build next

**Say:**
- "**Limitations:** it's synthetic data from one simulator, so the ACH dependence won't
  carry over as is. And 5.5 days of training history is short, so 7-day windows are
  never full in training."
- "**For production:** online scoring that reuses this exact feature code, checked by
  replaying the test period. A per-day alert queue, because weekends overflow a fixed
  threshold. Monitoring of alert rates and of the precision of worked alerts."
- "**For the model:** windowed counterparty counts, the runner-up, and a graph neural
  network compared on this same split."

**Close:** "The core lesson: the honest number comes from getting the time
handling right. That means leak-free features, a real future test period, and decisions
you can't bend after the fact."
