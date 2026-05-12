---
id: discounted-cash-flow
title: Discounted Cash Flow Valuation
topic: finance/valuation
triggers: DCF, discounted cash flow, intrinsic value, present value of cash flows, enterprise value, terminal value
summary: Value an asset or business as the present value of its future free cash flows discounted at the cost of capital.
---

# Discounted Cash Flow Valuation

DCF estimates intrinsic value by projecting future free cash flows and discounting them back to the present using a required rate of return.

## Formula

Enterprise Value = Σ (FCF_t / (1 + r)^t) + Terminal Value / (1 + r)^N

where:
- FCF_t = free cash flow in year t
- r = discount rate (typically WACC)
- N = explicit forecast horizon (commonly 5–10 years)

## Free cash flow

FCF = EBIT × (1 − tax) + D&A − CapEx − ΔWorking Capital

## Terminal value

Two common approaches:
- Gordon growth: TV_N = FCF_{N+1} / (r − g), where g is the perpetual growth rate (must be ≤ long-term GDP growth)
- Exit multiple: TV_N = EBITDA_N × industry multiple

## When to use

- Stable, cash-generative businesses with predictable trajectories
- Investment decisions requiring an intrinsic-value reference
- Cross-checking against market multiples

## Pitfalls

- Garbage in, garbage out: small changes in r or g move valuation significantly
- Terminal value typically dominates (60–80% of total) — sensitivity-test it
- Mismatch between cash flow type (FCFF vs FCFE) and discount rate (WACC vs cost of equity)
- Optimistic growth rates that exceed plausible GDP growth indefinitely