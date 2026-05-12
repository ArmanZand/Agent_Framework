---
id: sensitivity-analysis
title: Sensitivity Analysis
topic: analysis/risk
triggers: sensitivity analysis, what-if analysis, tornado chart, scenario testing, single variable shock
summary: Vary one input at a time and observe the effect on an output. Identifies which assumptions drive the result.
---

# Sensitivity Analysis

A discipline for understanding how robust a model output is to its inputs. Not a forecast — a stress test.

## Method

1. Identify the output variable (e.g. NPV, IRR, EBITDA)
2. Identify input drivers (revenue growth, margin, discount rate, etc.)
3. Vary each input across a plausible range (e.g. ±10%, ±20%) holding others constant
4. Tabulate or plot the output response
5. Rank inputs by output sensitivity (tornado chart)

## When to use

- After any DCF or financial model — never present a point estimate alone
- Investment committee material: which assumptions need to be defended
- Risk identification: drivers ranked by impact, not probability
- Negotiation prep (M&A, pricing): know your walk-away points across scenarios

## Pitfalls

- One-at-a-time hides interactions; pair with scenario analysis (combined plausible states) and Monte Carlo (full distributions) when stakes are high
- Choosing arbitrary ±10% shocks instead of distributions grounded in historical data
- Sensitivity tables on irrelevant variables — focus on those that meaningfully move the output
- Symmetry assumption (downside = upside) is often wrong; consider asymmetric ranges
- Presenting sensitivity without a clear recommendation; the analyst's job is to draw a conclusion