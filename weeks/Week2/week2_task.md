# Week 2 — Validation and the leakage lab

**Level: Specified.** The model is chosen for you and held fixed. Exactly one
thing changes between the two runs you will make, and it is one word.

**Project time: ~6 h per pair.** Swap driver and navigator every 30 minutes.

**No planted bugs this week.** `week2.ipynb` is correct. The wrong answer is
something you will produce deliberately, by choosing it.

---

## Before you open the notebook

Write down, in one sentence each:

1. You have ~640 laboratory measurements of penicillin titer, from 30
   fermentation runs. How many independent data points do you have? Justify the
   number you wrote.
2. You are about to hold out 30% of the data to test a model. What should the
   30% be 30% *of*?
3. Predict the R² you expect. Commit to a number before you run anything.

---

## The task

`bioml.split(df, how=...)` takes one of `"random"`, `"batchwise"` or
`"regime"`. **It has no default.** Both of the first two are one word, neither
is recommended anywhere in the API or in this task sheet, and you must choose.

Run the identical model under both, and account for the difference.

### 1. Two protocols, one model

Fit the provided ridge pipeline using `how="random"`, then using
`how="batchwise"`. Report for each: R², RMSE, and the number of batches that
appear in both the training and the test set.

That third number is not decoration. It is the mechanism, expressed as a count.

### 2. Cross-validation

Repeat with `bioml.cv_score(..., how=...)`. Report the mean **and the spread**
across folds for both protocols.

The spread is the more interesting of the two numbers and most reports omit it.
With 30 batches, the fold-to-fold variation is often larger than the difference
between two candidate models. State what that implies for a report that
compares two models on their mean scores alone.

### 3. Read the source

Open `bioml/splits.py` and read `split()`.

**In exactly two sentences, explain why `how="random"` inflates the score on
this data.** A complete answer names the unit of independence and says what a
random split does to it. Two sentences — this is a test of whether you can say
it, not of how much you can write.

### 4. Which one is right?

Trick question, and it is the point of the week. State what question each
protocol answers, and name a situation in which each is the correct choice. If
you conclude that one of them is never correct for this data, defend that.

---

## Deliverable: leakage lab report

Two to three pages:

- The four results above, in one table.
- The two-sentence explanation from part 3.
- Your pre-run R² prediction, and what it was wrong about.
- One paragraph: you are handed a model report claiming R² = 0.95 for a titer
  soft sensor, with no description of how the data was split. What do you ask
  first, and what would make you reject the number outright?

---

## How this is assessed

On the argument, not the numbers. "This model cannot be validated on this data,
and here is why" is a complete answer if it is defended.

The oral-defence scope applies as it did in Week 1. You called `split()` and
`cv_score()`; their source is on disk and you are expected to have read it.

---

## Suggested time split

| | |
|---|---|
| Pre-run predictions | 0.5 h |
| Run both protocols, both splits and CV | 1.5 h |
| Read `bioml/splits.py`, write the two sentences | 1.0 h |
| Part 4: which one is right, and when | 1.0 h |
| Write the report | 2.0 h |
