# High-Performance Intraday Lead-Lag Statistical Arbitrage Pipeline

An institutional-grade, high-frequency pairs trading backtest engine optimized for memory efficiency and execution speed (*C-Speed* processing). The framework mitigates structural backtesting biases and evaluates statistical lead-lag anomalies across the Nasdaq-100 constituents using QQQ as the market baseline.

## Key Architectural Implementations

* **Survivorship Bias Mitigation:** Dynamic historical universe depuration via timezone-aware exclusion windows.
* **Look-Ahead Bias Elimination:** Strict execution of rolling multi-day calculations restricted to trailing market closures ($T-1$).
* **Memory & Computational Optimization:** Vectorized analytical scaling utilizing `float32` continuous memory blocks and `category` formatting, executing millions of intra-minute operations through pure NumPy matrix arrays.
* **Risk Management:** Enforces fixed-nominal capital allocation sizing and compulsory 15:55 PM intraday position liquidation to eliminate overnight gap exposure.

## Repository Structure

```text
├── docs/                     # Academic and research foundations
│   └── Quantitative_Report.pdf      # Theoretical framework paper
├── Backtest_Engine.py        # Microstructure simulation & KPI suite
├── requirements.txt          # System dependencies
└── .gitignore                # Version control exclusions
