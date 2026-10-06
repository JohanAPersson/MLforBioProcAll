# Week 1 — Bioprocess data and the ML framing

**Level: Specified.** Every modelling decision this week is made for you and
stated as a choice, with the alternative named. Your job is to understand why
each was made, and to find the two places where the notebook gets it wrong.

**Project time: ~6 h per pair.** One keyboard. **Swap driver and navigator every
30 minutes** — this is a rule of the course, not a suggestion.

---

## Before you open the notebook

Read `prepare_indpensim.py`, sections 1–4. It is the answer to "what was done
to my data before I got it, and why". It is 15 minutes and it will save you
more than that.

Then answer, in writing, before running anything:

1. The source file is 2.57 GB and contains 100 fermentation runs. Which of
   those two numbers constrains how complex a model you can fit, and why?
2. You are going to build a sensor that estimates penicillin titer. Name three
   signals you would expect to carry information about it, and the physical
   reason for each.

Keep these. Deliverable (b) asks you to revisit them.

---

## The notebook

`week1.ipynb` loads batches 1–30, profiles them, and computes three derived
rates: specific growth rate µ, specific productivity q_P, and specific oxygen
uptake q_O₂.

> **This notebook runs. It produces the wrong answer. There are exactly two
> errors, both in the reasoning, neither in the syntax.**

Nothing raises an exception. Nothing prints a warning. Both errors are in
`derive_rates()`, and both concern quantities you already understand better
than you understand Python.

Two hints, and they are the only two you get:

- **Check the units.** Write out the units of every term on both sides of each
  rate definition and see whether they cancel. One of the two errors is
  visible from dimensional analysis alone, without running anything.
- **Check the magnitudes.** You know roughly how fast a filamentous fungus
  grows. One of the numbers this notebook prints is impossible, and you are
  equipped to say so without looking anything up.

Use `predict-before-run`: before executing a cell, write one falsifiable
sentence about what you expect it to print. A prediction you cannot be wrong
about is not a prediction.

---

## Deliverables

### (a) Column-role classification

For every non-Raman column, state whether it would exist on a real 100,000 L
production fermenter at hour 100, and justify it in one line.

`column_roles.csv` ships with the data and contains one answer. **Do not copy
it.** Produce yours first, then compare, and write a short note on every column
where you disagree — including any where you think the shipped classification is
wrong. Disagreeing and defending it scores better than agreeing without a reason.

Two columns to pay particular attention to: `Penicillin concentration(P:g/L)` and
`Substrate concentration(S:g/L)`. Both are in the file. Say what would happen to
a titer soft sensor that used the first one as an input, and be specific about
the number you would see.

### (b) Data profile and mechanistic hypothesis

A short written profile of batches 1–30: how long the batches run, how complete
each column is, which columns never move, and what the trajectories look like.
Then two or three sentences naming the variables you expect to drive final
titer, the physical reason for each, and what in the plots supports or
undermines it.

This will be held against your Week 3 and Week 4 models. It is allowed to turn
out wrong. A hypothesis that survives contact with data unchanged usually was
not falsifiable in the first place.

### (c) The two errors

For each: which line, what is wrong with the reasoning, what the correct version
is, and — this is the part that carries the marks — **how large the error was**.
Give the factor, and say what in the data determines it.

---

## How this is assessed

Not on accuracy. A report that concludes "these rates cannot be computed
reliably from this data, and here is why" can score full marks.

You may use any AI tool you like. Grading is on oral defence:

> You must be able to explain what any function you called does, why you chose
> it over the alternative, and what would change if you had chosen differently.
> The source of everything you were given is published. "I don't know what it
> does" is not available to you.

One of you will be cold-called at the weekly checkpoint and questioned by a
partner primed to disagree with you.

---

## Suggested time split

| | |
|---|---|
| Read `prepare_indpensim.py`, write the pre-run answers | 0.5 h |
| Run the notebook, predict-before-run throughout | 1.0 h |
| Deliverable (a), column roles | 1.5 h |
| Find and characterise the two errors | 1.5 h |
| Deliverable (b), profile and hypothesis | 1.0 h |
| Write up | 0.5 h |

If you are more than an hour past this on the error hunt, stop and ask. The
errors are conceptual, not hidden — being stuck on them is a signal about the
framing, not about you.
