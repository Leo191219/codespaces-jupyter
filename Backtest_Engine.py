import os # For reading, verifying, and clearing information.
import pandas as pd # Tabular data manipulation (DataFrame), time handling, sorting, transforming, and CSV exporting.
import numpy as np # Handles vectors and matrices in C. Compresses data, provides extreme speed, filters values, and calculates square roots.
import matplotlib.pyplot as plt # Plotting.
import seaborn as sns # Improves plot aesthetics.
import gc # Destroys unused information to save RAM. Cleaned up after each day.
from typing import List, Dict, Tuple, Optional # Corporate Type Hinting.
import gdown # Interacts directly with the Google Drive API.

def download_data(file_id: str, file_name: str) -> None:
    # Data Integrity Check: Drops corrupted or empty downloads (under 1KB) 
    # to prevent silent crashes in the backtest engine.
    if os.path.exists(file_name) and os.path.getsize(file_name) < 1024:
        print(f"Warning: Corrupted file detected -> {file_name}. Deleting...")
        os.remove(file_name)

    # Efficient Ingestion Flow: Skips download if the pristine Parquet file 
    # already exists locally, saving bandwidth and time.
    if not os.path.exists(file_name):
        print(f"Fetching {file_name} from cloud storage...")
        try:
            gdown.download(id=file_id, output=file_name, quiet=False)
        except Exception as e:
            # Fault Tolerance: Safely halts the pipeline if critical data is missing, 
            # preventing hallucinated calculations.
            print(f"Critical network failure downloading {file_name}: {e}")
            raise RuntimeError(f"Execution aborted: Missing required dataset -> {file_name}") from e
    else:
        # Success state: Confirms file readiness and footprint
        size_mb = os.path.getsize(file_name) / (1024**2)
        print(f"System ready: {file_name} validated ({size_mb:.2f} MB).")

# DATA INGESTION PIPELINE
# Cloud drive unique identifiers
QQQ_ID = "1bePMC7Mh3L_xH1YZJ1t7L7GXE7WL8PQh"
NASDAQ_ID = "1c8ctKH_qV8QrcFn2DUanab7U23_k3Pkw"

# Execute controlled downloads for the required time-series
download_data(QQQ_ID, "QQQ_1m.parquet")
download_data(NASDAQ_ID, "nasdaq100_with_meta.parquet")

# Corporate visual environment setup
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# Restricts assets dynamically to mitigate historical survivorship bias
RESTRICTED_TICKERS: Dict[str, List[str]] = {
    '2021-01-01/2021-12-31': ['ABNB', 'APP', 'ARM', 'AXON', 'CCEP', 'CEG', 'CRWD', 'DASH', 'DDOG', 'ENPH', 'FER', 'FTNT', 'GEHC', 'GFS', 'INSM', 'LCID', 'LIN', 'MDB', 'META', 'MPWR', 'MSTR', 'MTCH', 'NTES', 'ON', 'PLTR', 'RIVN', 'SHOP', 'STX', 'TCOM', 'TTD', 'WBD', 'WDC', 'WMT', 'ZS'],
    '2022-01-01/2022-12-31': ['ALNY', 'APP', 'ARM', 'AXON', 'CEG', 'DASH', 'FER', 'GEHC', 'GFS', 'INSM', 'LIN', 'MDB', 'META', 'MPWR', 'MSTR', 'NTES', 'ON', 'PLTR', 'RIVN', 'SHOP', 'STX', 'TCOM', 'TTD', 'WBD', 'WDC', 'WMT', 'ZS'],
    '2023-01-01/2023-12-31': ['ALNY', 'APP', 'ARM', 'AXON', 'CCEP', 'DASH', 'FER', 'GEHC', 'INSM', 'LIN', 'MDB', 'MPWR', 'MSTR', 'ON', 'PLTR', 'SHOP', 'STX', 'TTD', 'WDC', 'WMT', 'ZS'],
    '2024-01-01/2024-12-31': ['ALNY', 'APP', 'ARM', 'ATVI', 'AXON', 'BIIB', 'CERN', 'CHKP', 'DOCU', 'EBAY', 'ENPH', 'FB', 'FER', 'FISV', 'FOX', 'FOXA', 'INSM', 'JD', 'LCID', 'LIN', 'MPWR', 'MSTR', 'MXIM', 'NTES', 'OKTA', 'PLTR', 'PTON', 'RIVN', 'SGEN', 'SHOP', 'SPLK', 'STX', 'SWKS', 'TCOM', 'TTD', 'VRSN', 'WDC', 'WMT', 'XLNX', 'ZM'],
    '2025-01-01/2025-12-21': ['ALGN', 'ATVI', 'BIIB', 'CERN', 'CHKP', 'DOCU', 'DLTR', 'EBAY', 'ENPH', 'FB', 'FER', 'FISV', 'FOX', 'FOXA', 'INSM', 'JD', 'LCID', 'MNST', 'MPWR', 'MTCH', 'MXIM', 'NTES', 'OKTA', 'ON', 'PTON', 'RIVN', 'SGEN', 'SHOP', 'SIRI', 'SPLK', 'STX', 'SWKS', 'TCOM', 'TTD', 'VRSN', 'WBA', 'WDC', 'WMT', 'XLNX', 'ZM'],
    '2025-12-22/2026-01-19': ['ALGN', 'ATVI', 'BIIB', 'CDW', 'CERN', 'CHKP', 'DOCU', 'DLTR', 'EBAY', 'ENPH', 'FB', 'FISV', 'FOX', 'FOXA', 'GFS', 'JD', 'LCID', 'LULU', 'MRNA', 'MTCH', 'MXIM', 'NTES', 'OKTA', 'ON', 'PTON', 'RIVN', 'SGEN', 'SHOP', 'SIRI', 'SPLK', 'SWKS', 'TCOM', 'TTD', 'VRSN', 'WBA', 'WMT', 'XLNX', 'ZM'],
    '2026-01-20/2026-12-31': ['ALGN', 'ATVI', 'AZN', 'BIIB', 'CDW', 'CERN', 'CHKP', 'DOCU', 'DLTR', 'EBAY', 'ENPH', 'FB', 'FISV', 'FOX', 'FOXA', 'GFS', 'JD', 'LCID', 'LULU', 'MRNA', 'MTCH', 'MXIM', 'NTES', 'OKTA', 'ON', 'PTON', 'RIVN', 'SGEN', 'SHOP', 'SIRI', 'SPLK', 'SWKS', 'TCOM', 'TTD', 'VRSN', 'WBA', 'XLNX', 'ZM']
}

class UniverseManager:
    # Pre-computes exact timezone-aware intervals to filter out non-tradable assets
    def __init__(self, restricted_dict: Dict[str, List[str]]):
        self.restricted_periods: List[Tuple[pd.Timestamp, pd.Timestamp, set]] = []
        for period, tickers in restricted_dict.items():
            start_str, end_str = period.split('/')
            start_dt = pd.to_datetime(start_str).tz_localize('America/New_York')
            end_dt = pd.to_datetime(end_str).tz_localize('America/New_York') + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            self.restricted_periods.append((start_dt, end_dt, set(tickers)))

    def filter_universe(self, date: pd.Timestamp, tickers: List[str]) -> List[str]:
        # Evaluates the active timestamp and returns only legally tradable components
        restricted_set = set()
        for start, end, t_set in self.restricted_periods:
            if start <= date <= end:
                restricted_set.update(t_set)
        return [t for t in tickers if t not in restricted_set]


class DataEngine:
    def __init__(self, nasdaq_path: str, qqq_path: str):
        print("Mapping sector metadata and scaling attributes...")
        self.metadata: pd.DataFrame = pd.read_parquet(nasdaq_path, columns=['symbol', 'sic_description', 'market_cap']).drop_duplicates(subset=['symbol'])

        print("Ingesting optimized time series...")
        self.nasdaq_1m: pd.DataFrame = pd.read_parquet(nasdaq_path, columns=['date', 'symbol', 'close'])
        self.qqq_1m: pd.DataFrame = pd.read_parquet(qqq_path, columns=['date', 'close'])

        print("Synchronizing New York timestamps across all datasets...")
        self.nasdaq_1m['date'] = pd.to_datetime(self.nasdaq_1m['date'])
        if self.nasdaq_1m['date'].dt.tz is None:
            self.nasdaq_1m['date'] = self.nasdaq_1m['date'].dt.tz_localize('America/New_York')
        self.nasdaq_1m.set_index('date', inplace=True)

        if 'date' in self.qqq_1m.columns:
            self.qqq_1m['date'] = pd.to_datetime(self.qqq_1m['date'])
            if self.qqq_1m['date'].dt.tz is None:
                self.qqq_1m['date'] = self.qqq_1m['date'].dt.tz_localize('America/New_York')
            self.qqq_1m.set_index('date', inplace=True)
        else:
            if self.qqq_1m.index.tz is None:
                self.qqq_1m.index = pd.to_datetime(self.qqq_1m.index).tz_localize('America/New_York')

        print("Executing analytical RAM compression (Float32 & Category matrices)...")
        self.nasdaq_1m['symbol'] = self.nasdaq_1m['symbol'].astype('category')
        self.nasdaq_1m['close'] = self.nasdaq_1m['close'].astype(np.float32)
        if 'close' in self.qqq_1m.columns:
            self.qqq_1m['close'] = self.qqq_1m['close'].astype(np.float32)

        print("Structuring daily baselines for correlation mapping...")
        self.daily_close: pd.DataFrame = self.nasdaq_1m.groupby([self.nasdaq_1m.index.date, 'symbol'], observed=True)['close'].last().unstack()
        self.daily_close.index = pd.to_datetime(self.daily_close.index)
        self.daily_returns: pd.DataFrame = self.daily_close.ffill().pct_change().astype(np.float32)

        self.qqq_daily_close: pd.Series = self.qqq_1m.groupby(self.qqq_1m.index.date)['close'].last()
        self.qqq_daily_close.index = pd.to_datetime(self.qqq_daily_close.index)
        print("Data architecture consolidated successfully.")

    def generate_top_50_pairs(self, target_date: pd.Timestamp, universe: UniverseManager) -> List[Tuple[str, str]]:
        # Extracts a strict 60-day lagging window to calculate Pearson correlations without look-ahead bias
        local_date = target_date.tz_localize(None)
        historical_returns = self.daily_returns.loc[:local_date - pd.Timedelta(days=1)].tail(60)
        
        tradable_tickers = universe.filter_universe(target_date, list(historical_returns.columns))
        active_columns = [c for c in tradable_tickers if c in historical_returns.columns]
        
        historical_returns = historical_returns[active_columns].dropna(how='all', axis=1)

        if historical_returns.empty or len(historical_returns.columns) < 2:
            return []

        correlation_matrix = historical_returns.corr()
        sector_map = self.metadata.set_index('symbol')['sic_description'].to_dict()
        mcap_map = self.metadata.set_index('symbol')['market_cap'].to_dict()

        candidate_pairs = []
        tickers = list(correlation_matrix.columns)

        # Iterates through unique combinations to match intra-sector assets
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                
                if sector_map.get(t1) == sector_map.get(t2) and sector_map.get(t1) is not None:
                    rho = correlation_matrix.at[t1, t2]
                    
                    if not np.isnan(rho):
                        # Leader designation is strictly dictated by market capitalization
                        leader = t1 if mcap_map.get(t1, 0) > mcap_map.get(t2, 0) else t2
                        follower = t2 if leader == t1 else t1
                        candidate_pairs.append((rho, leader, follower))

        # Sorts matches descending by Pearson rho and trims to the optimal Top 50 subset
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)
        return [(item[1], item[2]) for item in candidate_pairs[:50]]


class SimulationEngine:
    def __init__(self, data_engine: DataEngine, initial_capital: float = 1000000.0):
        self.engine = data_engine
        self.initial_capital = initial_capital
        self.trades: List[Dict] = []

    def run_backtest(self, universe: UniverseManager) -> pd.DataFrame:
        print("Engaging intraday microstructure simulation...")
        
        all_days = sorted(self.engine.daily_returns.index.unique())
        simulation_days = set([d.date() for d in all_days if d >= all_days[60]])

        self.engine.nasdaq_1m['date_only'] = self.engine.nasdaq_1m.index.date

        # Macro loop: Daily grouping to optimize memory footprint
        for date_key, daily_data in self.engine.nasdaq_1m.groupby('date_only'):
            if date_key not in simulation_days:
                continue

            target_timestamp = pd.Timestamp(date_key).tz_localize('America/New_York')
            selected_pairs = self.engine.generate_top_50_pairs(target_timestamp, universe)
            
            if not selected_pairs:
                continue

            active_tickers = set([t for pair in selected_pairs for t in pair])
            filtered_data = daily_data[daily_data['symbol'].isin(active_tickers)]
            
            if filtered_data.empty:
                continue

            # Calculates intraday moving averages using the pre-market buffer
            minute_prices = filtered_data.pivot(columns='symbol', values='close').dropna(how='all', axis=1).ffill().bfill()
            minute_smas = minute_prices.rolling(window=15, min_periods=1).mean()

            market_open = pd.Timestamp.combine(date_key, pd.Timestamp('09:30:00').time()).tz_localize('America/New_York')
            forced_exit_time = pd.Timestamp.combine(date_key, pd.Timestamp('15:55:00').time()).tz_localize('America/New_York')
            market_close = pd.Timestamp.combine(date_key, pd.Timestamp('16:00:00').time()).tz_localize('America/New_York')

            # Crops the matrices to official market hours
            session_prices = minute_prices.loc[market_open:market_close]
            session_smas = minute_smas.loc[market_open:market_close]

            if session_prices.empty:
                continue

            # Memory allocation to contiguous NumPy arrays for C-speed iteration
            time_ticks = session_prices.index
            prices_matrix = session_prices.values
            smas_matrix = session_smas.values

            columns_list = list(session_prices.columns)
            ticker_to_idx = {t: i for i, t in enumerate(columns_list)}

            pair_indices = []
            for pair in selected_pairs:
                leader, follower = pair
                if leader in ticker_to_idx and follower in ticker_to_idx:
                    pair_indices.append((ticker_to_idx[leader], ticker_to_idx[follower], pair))

            position_state = {pair: None for pair in selected_pairs}
            position_details = {pair: {} for pair in selected_pairs}

            # Micro loop: Minute-by-minute order execution
            for k in range(len(time_ticks)):
                tick = time_ticks[k]
                is_forced_exit = (tick >= forced_exit_time)

                for l_idx, f_idx, pair in pair_indices:
                    price_l = prices_matrix[k, l_idx]
                    price_f = prices_matrix[k, f_idx]
                    sma_l = smas_matrix[k, l_idx]
                    sma_f = smas_matrix[k, f_idx]

                    state = position_state[pair]

                    # Risk mitigation: Overnight exposure is strictly prohibited
                    if is_forced_exit and state is not None:
                        self._close_position(pair, tick, price_f, position_details[pair], 'Forced Close (15:55)', state)
                        position_state[pair] = None
                        continue

                    if state == 'LONG':
                        if price_l < sma_l:
                            self._close_position(pair, tick, price_f, position_details[pair], 'Leader broke base SMA', 'LONG')
                            position_state[pair] = None
                    elif state == 'SHORT':
                        if price_l > sma_l:
                            self._close_position(pair, tick, price_f, position_details[pair], 'Leader broke base SMA', 'SHORT')
                            position_state[pair] = None
                            
                    elif state is None and not is_forced_exit:
                        if price_f <= 10.0 or price_l <= 10.0:  # Enforces minimum liquidity standards
                            continue

                        # Core strategy: Asymmetric standard deviation triggers based on lead-lag cointegration
                        if price_l > sma_l * 1.006 and price_f < sma_f * 1.002:
                            qty = 100000.0 / price_f  # Capital allocation: $100k nominal per trade
                            position_state[pair] = 'LONG'
                            position_details[pair] = {
                                'entry_time': tick, 'entry_price': price_f, 'quantity': qty,
                                'entry_tx_cost': qty * 0.0035, 'reason_entry': 'Bullish Lead-Lag Breakout'
                            }
                        elif price_l < sma_l * 0.994 and price_f < sma_f * 0.998:
                            qty = 100000.0 / price_f
                            position_state[pair] = 'SHORT'
                            position_details[pair] = {
                                'entry_time': tick, 'entry_price': price_f, 'quantity': qty,
                                'entry_tx_cost': qty * 0.0035, 'reason_entry': 'Bearish Lead-Lag Breakout'
                            }

            # Explicit RAM flushing to prevent cumulative memory leaks across simulated days
            del minute_prices, minute_smas, session_prices, session_smas, prices_matrix, smas_matrix
            gc.collect()

        return pd.DataFrame(self.trades)

    def _close_position(self, pair: Tuple[str, str], tick: pd.Timestamp, price_f: float, details: Dict, reason_exit: str, side: str) -> None:
        qty = details['quantity']
        entry_p = details['entry_price']
        total_tx = details['entry_tx_cost'] + (qty * 0.0035)  # Applies institutional per-share friction costs

        # Asymmetric PnL resolution mapping
        if side == 'LONG':
            pnl_cash = (price_f - entry_p) * qty - total_tx
            pnl_return = ((price_f - entry_p) / entry_p) - (total_tx / 100000.0)
        else:
            pnl_cash = (entry_p - price_f) * qty - total_tx
            pnl_return = ((entry_p - price_f) / entry_p) - (total_tx / 100000.0)

        # Commits transactional receipt to the immutable ledger
        self.trades.append({
            'leader': pair[0], 'follower': pair[1], 'entry time': details['entry_time'], 'exit time': tick,
            'entry price': entry_p, 'exit price': price_f, 'quantity': qty, 'side': side,
            'reason for entry/exit': f"IN: {details['reason_entry']} | OUT: {reason_exit}",
            'PnL in dollars': pnl_cash, 'PnL in percentage terms': pnl_return * 100, 'transaction costs': total_tx
        })


# ==============================================================================
# EXECUTION & KPI REPORTING SUITE
# ==============================================================================
if __name__ == "__main__":
    universe = UniverseManager(RESTRICTED_TICKERS)
    engine = DataEngine(nasdaq_path="nasdaq100_with_meta.parquet", qqq_path="QQQ_1m.parquet")
    simulator = SimulationEngine(engine, initial_capital=1000000.0)

    trade_log = simulator.run_backtest(universe)

    if not trade_log.empty:
        # Resolves daily PnL vectors into an absolute equity curve
        trade_log['date'] = pd.to_datetime(trade_log['exit time']).dt.tz_localize(None).dt.normalize()
        daily_pnl = trade_log.groupby('date')['PnL in dollars'].sum()

        sim_days = sorted(daily_pnl.index)
        strategy_equity = pd.Series(index=sim_days, dtype=float)

        current_cap = simulator.initial_capital
        for d in sim_days:
            current_cap += daily_pnl.get(d, 0.0)
            strategy_equity[d] = current_cap

        strategy_returns = strategy_equity.pct_change().fillna(0)

        # Models the QQQ baseline index for apples-to-apples benchmarking
        bench_prices = engine.qqq_daily_close.loc[min(sim_days):max(sim_days)]
        benchmark_equity = (bench_prices / bench_prices.iloc[0]) * simulator.initial_capital
        benchmark_returns = benchmark_equity.pct_change().fillna(0)

        def get_advanced_kpis(equity: pd.Series, returns: pd.Series) -> Tuple[float, float, float, float, float, int, pd.Series]:
            """Resolves institutional risk metrics based on Modern Portfolio Theory."""
            total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            ann_ret = ((equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1) * 100
            ann_vol = returns.std() * np.sqrt(252) * 100
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0

            roll_max = equity.cummax()
            dd = (equity - roll_max) / roll_max * 100

            running_max = -1.0
            peak_date = equity.index[0]
            max_duration = 0
            for date, val in equity.items():
                if val >= running_max:
                    running_max = val
                    peak_date = date
                else:
                    duration = (date - peak_date).days
                    if duration > max_duration:
                        max_duration = duration

            return total_ret, ann_ret, ann_vol, sharpe, dd.min(), max_duration, dd

        strat_tot, strat_ann, strat_vol, strat_sh, strat_mdd, strat_dur, strat_dd = get_advanced_kpis(strategy_equity, strategy_returns)
        bench_tot, bench_ann, bench_vol, bench_sh, bench_mdd, bench_dur, bench_dd = get_advanced_kpis(benchmark_equity, benchmark_returns)

        performance_table = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Max Drawdown Duration'],
            'Strategy (Lead-Lag)': [f"{strat_tot:.2f}%", f"{strat_ann:.2f}%", f"{strat_vol:.2f}%", f"{strat_sh:.2f}", f"{strat_mdd:.2f}%", f"{strat_dur} Days"],
            'Benchmark (QQQ)': [f"{bench_tot:.2f}%", f"{bench_ann:.2f}%", f"{bench_vol:.2f}%", f"{bench_sh:.2f}", f"{bench_mdd:.2f}%", f"{bench_dur} Days"]
        })

        print("\n" + "="*60)
        print("             INDEPENDENT PERFORMANCE REPORT TABLE             ")
        print("="*60)
        print(performance_table.to_string(index=False))
        print("="*60 + "\n")

        # Visualizations
        plt.figure()
        plt.plot(strategy_equity.index, strategy_equity.values, label='Strategy (Lead-Lag ML Baseline)', color='navy', lw=2)
        plt.plot(benchmark_equity.index, benchmark_equity.values, label='Benchmark (QQQ Buy & Hold)', color='crimson', linestyle='--', lw=1.5)
        plt.yscale('log')
        plt.title('Figure 1: Cumulative Equity Curves (Logarithmic Scale)', fontsize=14, fontweight='bold')
        plt.xlabel('Timeline', fontsize=12, fontweight='bold')
        plt.ylabel('Portfolio Valuation ($) - Log Scale', fontsize=12, fontweight='bold')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('equity_curve_log.png', dpi=300)
        plt.show()

        plt.figure()
        plt.fill_between(strat_dd.index, strat_dd.values, 0, label='Strategy Drawdown Profile', color='navy', alpha=0.3)
        plt.fill_between(bench_dd.index, bench_dd.values, 0, label='Benchmark QQQ Drawdown Profile', color='crimson', alpha=0.15)
        plt.title('Figure 2: Percentage Drawdown Profiles over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Timeline', fontsize=12, fontweight='bold')
        plt.ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig('drawdowns_percentage.png', dpi=300)
        plt.show()

        # CSV Logging
        equity_export = pd.DataFrame({'Strategy_Equity': strategy_equity, 'Benchmark_Equity': benchmark_equity})
        equity_export['Strategy_Equity'] = equity_export['Strategy_Equity'].ffill().fillna(simulator.initial_capital)
        equity_export.to_csv('equity_curve_output.csv')
        trade_log.to_csv('trade_log_output.csv', index=False)
        
        print("PROCESS SUCCESSFULLY COMPLETED!")
        print("Exported files:\n1. 'equity_curve_output.csv'\n2. 'trade_log_output.csv'")
    else:
        print("Critical anomaly: The backtest transactional ledger returned empty.")