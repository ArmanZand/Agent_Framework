---
id: internal-rate-of-return
title: Internal Rate of Return (IRR)
topic: finance/capital-budgeting
triggers: IRR, internal rate of return, hurdle rate comparison, project return, MIRR
summary: Discount rate that makes NPV equal to zero. Decision rule, accept if IRR exceeds the hurdle rate.
---

# Internal Rate of Return (IRR)

IRR is the discount rate at which a project's NPV equals zero — the breakeven cost of capital.

## Definition

Find r such that: 0 = −C_0 + Σ (CF_t / (1 + r)^t)

Solved iteratively; no closed form.

## Decision rule

Accept if IRR > hurdle rate (typically WACC plus a project-specific risk premium).

## When to use

- Quick comparison of project returns
- Communication with non-financial stakeholders (a "rate" is intuitive)
- Single-project go/no-go decisions with clean cash flow shape

## Pitfalls

- Multiple IRRs when cash flows change sign more than once; use NPV instead
- IRR can mislead when comparing mutually exclusive projects of different scales — a small high-IRR project may create less absolute value than a larger lower-IRR one
- Reinvestment assumption: IRR implicitly assumes intermediate cash flows are reinvested at the IRR itself, which is often unrealistic
- Modified IRR (MIRR) addresses the reinvestment problem
- When in doubt, prefer NPV