---
skill: portfolio-analysis
name: Portfolio Analysis
description: Evaluate a portfolio's composition, risk exposures, diversification quality, and performance attribution.
trigger: When the user asks you to analyse, review, or assess a portfolio, holdings, or allocation.
---

# Portfolio Analysis

## Step 1 — Inventory

Before analysis, establish what you are working with:
- List of positions with weights (% of portfolio)
- Asset class breakdown (equities, fixed income, cash, alternatives, real assets)
- Geographic breakdown
- Sector breakdown (for equity sleeves)
- Currency exposure

If this information is incomplete, ask for it before proceeding. Analysis on incomplete data produces misleading conclusions.

## Step 2 — Concentration check

- Single position > 10% of total portfolio: flag for justification.
- Single sector > 30% of equity sleeve: elevated sector risk.
- Single country > 40% of equity sleeve: elevated geographic concentration.
- Correlation: if two large positions are highly correlated, treat them as one effective exposure.

## Step 3 — Diversification quality

Diversification is not just count — it is independence of return drivers.

Ask:
- Do the positions react similarly to rising rates? Rising inflation? Recession?
- Is the portfolio long a single factor (e.g., all growth, all small-cap)?
- Are there natural hedges (e.g., commodity producer offsets commodity consumer)?

A 20-stock portfolio all in tech is less diversified than a 10-stock portfolio spanning sectors.

## Step 4 — Risk exposure summary

Compute or estimate:
- **Portfolio beta** (weighted average of position betas vs. benchmark)
- **Duration** (for bond holdings): sensitivity to 100bp rate move
- **Drawdown history** if available: maximum historical peak-to-trough
- **Volatility** (annualised standard deviation if return series available)

## Step 5 — Performance attribution (if return data available)

Decompose returns into:
- **Allocation effect**: did the weighting decisions add value vs. benchmark?
- **Selection effect**: did individual security picks outperform within each segment?
- **Currency effect**: how much did FX moves contribute?
- **Factor attribution**: growth vs. value, size, momentum, quality

## Step 6 — Stress testing

Test the portfolio against these scenarios:
- Equity market -30% (broad bear market)
- Interest rates +200bp (rate shock)
- Credit spreads widen 300bp (credit crisis)
- Commodity price shock ±40%
- Target currency depreciates 20%

Estimate portfolio-level impact for each.

## Output format

1. **Portfolio snapshot** — asset class, sector, geography table with weights
2. **Concentration flags** — list any positions or clusters exceeding thresholds
3. **Risk exposures** — beta, duration, key factor tilts
4. **Diversification verdict** — effective number of independent bets
5. **Stress test results** — estimated P&L impact per scenario
6. **Recommendations** — specific, actionable (not generic "diversify more")
