import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib
import backtrader as bt
import warnings
warnings.filterwarnings("ignore")
import glob

# Additional libraries
import ta

# Set paths
input_folder = "RawDataDoNotModify"  # Replace with your input folder path
output_folder = "output_folder_3"  # Replace with your output folder path
forward_testing_folder="Forward Testing"
os.makedirs(output_folder, exist_ok=True)

def load_data(file_path):
    """
    Load and preprocess data from the given CSV file.
    - Parses datetime columns.
    - Renames columns to a standardized schema.
    - Converts numeric columns to the correct type.
    - Fills missing values.
    """
    try:
        df = pd.read_csv(file_path)
        # print("Raw data head:")
        # print(df.head())

        # Handle the datetime column
        if 'Local time' in df.columns:  # Common column name for both formats
            df = df.rename(columns={'Local time': 'Datetime'})
            # Check for GMT format and handle
            if df['Datetime'].str.contains(r"GMT", regex=True).any():
                df['Datetime'] = df['Datetime'].str.replace(r" GMT[+-]\d{4}", "", regex=True)
            df['ds'] = pd.to_datetime(df['Datetime'], errors='coerce')  # Handle both formats
        elif 'Datetime' in df.columns:  # Generic datetime column fallback
            df['ds'] = pd.to_datetime(df['Datetime'], errors='coerce')

        # Remove timezone information if present
        if df['ds'].dt.tz is not None:
            df['ds'] = df['ds'].dt.tz_localize(None)

        # Ensure datetime column is valid
        df = df.dropna(subset=['ds'])
        df.sort_values('ds', inplace=True)

        # Handle numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'y']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(method='ffill')

        # Rename 'Close' or 'Adj Close' to 'y' (target variable)
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'y'})
        elif 'Adj Close' in df.columns:
            df = df.rename(columns={'Adj Close': 'y'})

        df.reset_index(drop=True, inplace=True)
        # print("Processed data head:")
        # print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()



def find_forward_test_file(forward_testing_folder, instrument_name):
    """
    Find the forward test file for a given instrument.
    The forward test file should have the instrument name before the first underscore in its filename.
    """
    instrument_prefix = instrument_name.split("_")[0]  # Extract the prefix before the first underscore
    matching_files = glob.glob(os.path.join(forward_testing_folder, f"{instrument_prefix}*.csv"))
    
    if not matching_files:
        raise FileNotFoundError(f"No forward testing file found for instrument: {instrument_name}")
    if len(matching_files) > 1:
        print(f"Warning: Multiple forward testing files found for {instrument_name}. Using the first one.")
    
    return matching_files[0]  # Return the first matching file


# Function to calculate technical indicators
def add_technical_indicators(df):
    """
    Compute various technical indicators (RSI, MACD, Stochastic, Bollinger Bands, ATR, Williams %R)
    and add them as columns to the DataFrame.
    """
    try:
        df['RSI'] = ta.momentum.RSIIndicator(close=df['y'], window=14).rsi()
        macd = ta.trend.MACD(close=df['y'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['y'], window=14)
        df['Stochastic'] = stoch.stoch()
        bollinger = ta.volatility.BollingerBands(close=df['y'], window=20, window_dev=2)
        df['BB_upper'] = bollinger.bollinger_hband()
        df['BB_lower'] = bollinger.bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['y'], window=14).average_true_range()
        df['Williams_%R'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['y'], lbp=14).williams_r()
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        return df
    except Exception as e:
        print(f"Error adding indicators: {e}")
        return pd.DataFrame()

# Backtrader trading strategy
class XGBoostStrategy(bt.Strategy):
    
    """
    A Backtrader strategy that:
    - Uses predictions from an XGBoost model to decide BUY/SELL actions.
    - Implements a simple take-profit and stop-loss mechanism.
    """
    params = dict(
        data_df=None,
        profit_target=0.02,
        stop_loss=0.01,
        position_size=0.10,
        instrument_output="",
    )

    def __init__(self):
        self.predictions = self.p.data_df.set_index('datetime')['xgb_prediction']
        self.predictions.index = pd.to_datetime(self.predictions.index)
        self.trade_id = 0
        self.trade_log = []

    def next(self):
        dt = self.data.datetime.datetime(0)
        if dt not in self.predictions.index:
            return

        predicted_price = self.predictions.loc[dt]
        actual_price = self.data.close[0]
        current_position = self.getposition()

        if not current_position:
            size = int((self.broker.getcash() * self.p.position_size) / actual_price)
            if size > 0:
                if predicted_price > actual_price:
                    self.buy(size=size)
                    action = 'BUY'
                elif predicted_price < actual_price:
                    self.sell(size=size)
                    action = 'SELL'
                else:
                    return

                self.trade_id += 1
                self.trade_log.append({
                    'Trade ID': self.trade_id,
                    'Entry Date': dt,
                    'Entry Price': actual_price,
                    'Size': size if action == 'BUY' else -size,
                    'Action': action
                })
        else:
            entry_trade = self.trade_log[-1]
            entry_price = entry_trade['Entry Price']
            if current_position.size > 0:
                if actual_price >= entry_price * (1 + self.p.profit_target) or \
                   actual_price <= entry_price * (1 - self.p.stop_loss):
                    profit = (actual_price - entry_price) * current_position.size
                    self.close()
                    self.trade_log[-1].update({
                        'Exit Date': dt,
                        'Exit Price': actual_price,
                        'Profit': profit,
                        'Duration (hours)': (dt - entry_trade['Entry Date']).total_seconds() / 3600,
                        'Result': 'WIN' if profit > 0 else 'LOSS'
                    })
            elif current_position.size < 0:
                if actual_price <= entry_price * (1 - self.p.profit_target) or \
                   actual_price >= entry_price * (1 + self.p.stop_loss):
                    profit = (entry_price - actual_price) * abs(current_position.size)
                    self.close()
                    self.trade_log[-1].update({
                        'Exit Date': dt,
                        'Exit Price': actual_price,
                        'Profit': profit,
                        'Duration (hours)': (dt - entry_trade['Entry Date']).total_seconds() / 3600,
                        'Result': 'WIN' if profit > 0 else 'LOSS'
                    })

    def stop(self):
        pass

# Function to calculate rolling Sharpe ratio
def calculate_rolling_sharpe_ratio(equity_curve, window=20):
    """
    Compute the rolling Sharpe ratio over a specified window of returns.
    """
    returns = equity_curve['cumulative'].pct_change()
    rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std()
    return rolling_sharpe

def prepare_backtest_data(df):
    """
    Rename columns and set the datetime index for backtesting.
    Backtrader expects certain column names: open, high, low, close, datetime as index.
    """
    try:
        required_cols = ['ds', 'Open', 'High', 'Low', 'y']
        if 'xgb_prediction' in df.columns:
            required_cols.append('xgb_prediction')
        df = df[required_cols].copy()
        df.rename(columns={'ds': 'datetime', 'y': 'close'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df[['open', 'high', 'low', 'close']] = df[['Open', 'High', 'Low', 'close']].fillna(method='ffill')
        return df
    except Exception as e:
        print(f"Error preparing backtest data: {e}")
        return pd.DataFrame()

def process_forward_test(forward_file, best_params, xgb_model, output_dir):
    """
    Forward test the best parameters on new unseen data.
    - Load forward test data.
    - Add indicators.
    - Predict with the trained model.
    - Run Backtrader simulation.
    - Save results (equity curve, sharpe ratio, drawdown, trades).
    """
    # Load and preprocess forward test data
    df = load_data(forward_file)
    if df.empty:
        raise ValueError(f"Forward test data is empty for file: {forward_file}")

    df = add_technical_indicators(df)
    if df.empty or not all(feature in df.columns for feature in [
        'RSI', 'MACD', 'MACD_Hist', 'Stochastic', 'BB_upper', 'BB_lower', 'ATR', 'Williams_%R']):
        raise ValueError(f"Technical indicators missing or incomplete in forward test data.")

    features = ['RSI', 'MACD', 'MACD_Hist', 'Stochastic', 'BB_upper', 'BB_lower', 'ATR', 'Williams_%R']
    #print("Forward Test Features Available:", df.columns)  # Debugging
    
    df['xgb_prediction'] = xgb_model.predict(df[features])
    forward_data = prepare_backtest_data(df)
    if forward_data.empty:
        raise ValueError("Forward test data preparation failed.")


    # Set up Backtrader
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000.0)
    data_feed = bt.feeds.PandasData(
        dataname=forward_data,
        open='open',
        high='high',
        low='low',
        close='close',
        volume=0,
        openinterest=-1
    )
    cerebro.adddata(data_feed)

    # Add the XGBoost strategy with best parameters
    cerebro.addstrategy(
        XGBoostStrategy,
        data_df=forward_data.reset_index(),
        position_size=best_params['position_size'],
        stop_loss=best_params['stop_loss'],
        profit_target=best_params['profit_target'],
        instrument_output=output_dir
    )

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')  # Added TimeReturn analyzer

    # Run the Backtrader engine
    results = cerebro.run()
    strat = results[0]

    # Extract equity curve
    timereturns = strat.analyzers.timereturn.get_analysis()
    equity_curve = pd.Series(timereturns).reset_index()
    equity_curve.columns = ['datetime', 'return']
    equity_curve.set_index('datetime', inplace=True)
    equity_curve['cumulative'] = (equity_curve['return'] + 1).cumprod()

    # Save equity curve plot
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve['cumulative'], label='Cumulative Returns')
    plt.title('Equity Curve (Forward Test)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    equity_curve_path = os.path.join(output_dir, "forward_equity_curve.png")
    plt.savefig(equity_curve_path)
    plt.close()
    print(f"Equity curve saved at: {equity_curve_path}")

    # Calculate and save rolling Sharpe ratio plot
    rolling_sharpe = equity_curve['cumulative'].pct_change().rolling(window=20).mean() / equity_curve['cumulative'].pct_change().rolling(window=20).std()
    plt.figure(figsize=(14, 7))
    plt.plot(rolling_sharpe, label='Rolling Sharpe Ratio')
    plt.title('Rolling Sharpe Ratio (Forward Test)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    sharpe_ratio_path = os.path.join(output_dir, "forward_rolling_sharpe_ratio.png")
    plt.savefig(sharpe_ratio_path)
    plt.close()
    print(f"Rolling Sharpe ratio saved at: {sharpe_ratio_path}")

    # Save equity curve to CSV
    equity_curve.to_csv(os.path.join(output_dir, "forward_equity_curve.csv"))

    # Save drawdown report
    drawdown = strat.analyzers.drawdown.get_analysis()
    drawdown_path = os.path.join(output_dir, "forward_drawdown_report.txt")
    with open(drawdown_path, "w") as f:
        f.write(str(drawdown))
    print(f"Drawdown report saved at: {drawdown_path}")

    # Save trade analysis
    trade_analysis = pd.DataFrame(strat.trade_log)
    trade_analysis_path = os.path.join(output_dir, "forward_trade_analysis.csv")
    with open(trade_analysis_path, "w") as f:
        f.write(str(trade_analysis))
    print(f"Trade analysis report saved at: {trade_analysis_path}")

    # Return final PnL
    final_pnl = cerebro.broker.getvalue() - 10000.0
    print(f"Final PnL (Forward Test): {final_pnl}")
    return final_pnl




def calculate_summary_metrics(equity_curve):
    """
    Calculate and return summary metrics like mean Sharpe ratio, maximum drawdown, and total return.
    """
    returns = equity_curve['cumulative'].pct_change()
    
    # Calculate mean Sharpe ratio (general practice)
    mean_sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0

    # Calculate maximum drawdown
    max_drawdown = equity_curve['cumulative'].div(equity_curve['cumulative'].cummax()).min() - 1

    # Calculate total return
    total_return = equity_curve['cumulative'].iloc[-1] - 1

    return mean_sharpe, max_drawdown, total_return


def process_instrument(file_path, output_folder, summary_results):
    """
    For each instrument (file):
    - Load and preprocess data.
    - Add technical indicators.
    - Split into train/test/backtest sets.
    - Train an XGBoost model on training data.
    - Generate predictions on test/backtest data.
    - Tune trading strategy parameters using backtest data.
    - Select best parameters and forward test them.
    - Save results and append summary metrics.
    """
    instrument_name = os.path.basename(file_path).split("_")[0]
    instrument_output = os.path.join(output_folder, instrument_name)
    os.makedirs(instrument_output, exist_ok=True)

    df = load_data(file_path)
    if df.empty:
        print(f"Skipping {instrument_name} due to insufficient or invalid data.")
        return

    df = add_technical_indicators(df)
    if df.empty:
        print(f"Skipping {instrument_name} due to indicator calculation errors.")
        return

    try:
        train_df = df.iloc[:-3000]
        test_df = df.iloc[-3000:-2000]
        backtest_df = df.iloc[-2000:]
    except Exception as e:
        print(f"Error splitting data: {e}")
        return

    features = ['RSI', 'MACD', 'MACD_Hist', 'Stochastic', 'BB_upper', 'BB_lower', 'ATR', 'Williams_%R']

    try:
        xgb_model = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.01)
        xgb_model.fit(train_df[features], train_df['y'])
        joblib.dump(xgb_model, os.path.join(instrument_output, "xgb_model.pkl"))
    except Exception as e:
        print(f"Error training XGBoost model: {e}")
        return

    try:
        test_df['xgb_prediction'] = xgb_model.predict(test_df[features])
        backtest_df['xgb_prediction'] = xgb_model.predict(backtest_df[features])

        # Save price prediction curves
        for data, label in [(test_df, "test"), (backtest_df, "backtest")]:
            plt.figure(figsize=(14, 7))
            plt.plot(data['ds'], data['y'], label='Actual Price')
            plt.plot(data['ds'], data['xgb_prediction'], label='Predicted Price')
            plt.title(f'Price Predictions ({label.capitalize()}) for {instrument_name}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(os.path.join(instrument_output, f"{label}_predictions.png"))
            plt.close()
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return

    param_grid = {
        'position_size': [0.05, 0.1, 0.15],
        'stop_loss': [0.01, 0.02, 0.03],
        'profit_target': [0.02, 0.03, 0.05]
    }

    best_pnl = -np.inf
    best_params = None

    backtest_data = prepare_backtest_data(backtest_df)
    if backtest_data.empty:
        print(f"Skipping {instrument_name} due to invalid backtest data.")
        return

    for ps in param_grid['position_size']:
        for sl in param_grid['stop_loss']:
            for pt in param_grid['profit_target']:
                try:
                    cerebro = bt.Cerebro()
                    cerebro.broker.setcash(10000.0)

                    data_feed = bt.feeds.PandasData(
                        dataname=backtest_data,
                        open='open',
                        high='high',
                        low='low',
                        close='close',
                        volume=0,
                        openinterest=-1
                    )
                    cerebro.adddata(data_feed)

                    strategy = cerebro.addstrategy(
                        XGBoostStrategy,
                        data_df=backtest_data.reset_index(),
                        position_size=ps,
                        stop_loss=sl,
                        profit_target=pt,
                        instrument_output=instrument_output
                    )

                    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
                    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
                    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')  # Added TimeReturn analyzer
                    results = cerebro.run()
                    strat = results[0]
                    pnl = cerebro.broker.getvalue() - 10000.0

                    if pnl > best_pnl:
                        best_pnl = pnl
                        best_params = {'position_size': ps, 'stop_loss': sl, 'profit_target': pt}

                        print(best_pnl)

                        # Save best trade log
                        best_trade_log = pd.DataFrame(strat.trade_log)
                        trade_log_path = os.path.join(instrument_output, "best_trade_log.csv")
                        best_trade_log.to_csv(trade_log_path, index=False)
                        print(f"Best trade log saved at {trade_log_path}")

                        # Extract equity curve
                        timereturns = results[0].analyzers.timereturn.get_analysis()
                        equity_curve = pd.Series(timereturns).reset_index()
                        equity_curve.columns = ['datetime', 'return']
                        equity_curve['datetime'] = pd.to_datetime(equity_curve['datetime'])
                        equity_curve.set_index('datetime', inplace=True)
                        equity_curve['cumulative'] = (equity_curve['return'] + 1).cumprod()

                        # Save equity curve
                        plt.figure(figsize=(14, 7))
                        plt.plot(equity_curve['cumulative'], label='Cumulative Returns')
                        plt.title(f'Best Equity Curve for {instrument_name}')
                        plt.xlabel('Date')
                        plt.ylabel('Cumulative Returns')
                        plt.legend()
                        plt.savefig(os.path.join(instrument_output, "best_equity_curve.png"))
                        plt.close()

                        # Save rolling Sharpe ratio
                        rolling_sharpe = calculate_rolling_sharpe_ratio(equity_curve)
                        plt.figure(figsize=(14, 7))
                        plt.plot(rolling_sharpe, label='Rolling Sharpe Ratio')
                        plt.title(f'Rolling Sharpe Ratio for {instrument_name}')
                        plt.xlabel('Date')
                        plt.ylabel('Sharpe Ratio')
                        plt.legend()
                        plt.savefig(os.path.join(instrument_output, "rolling_sharpe_ratio.png"))
                        plt.close()

                        # Calculate summary metrics for backtest
                        mean_sharpe, max_drawdown, total_return = calculate_summary_metrics(equity_curve)
                except Exception as e:
                    print(f"Error during grid search with params {ps, sl, pt}: {e}")

    # Forward testing
    try:
        forward_file = find_forward_test_file(forward_testing_folder, instrument_name)
        forward_pnl = process_forward_test(forward_file, best_params, xgb_model, instrument_output)
        forward_equity_curve = pd.read_csv(os.path.join(instrument_output, "forward_equity_curve.csv"))
        forward_mean_sharpe, forward_max_drawdown, forward_total_return = calculate_summary_metrics(forward_equity_curve)
    except:
        forward_pnl, forward_mean_sharpe, forward_max_drawdown, forward_total_return = 'N/A', 'N/A', 'N/A', 'N/A'

    summary_results.append({
        "Instrument": instrument_name,
        "Best PnL": best_pnl,
        "Best Params": best_params,
        "Mean Sharpe Ratio (Backtest)": mean_sharpe,
        "Maximum Drawdown (Backtest)": max_drawdown,
        "Total Return (Backtest)": total_return,
        "Forward PnL": forward_pnl,
        "Mean Sharpe Ratio (Forward)": forward_mean_sharpe,
        "Maximum Drawdown (Forward)": forward_max_drawdown,
        "Total Return (Forward)": forward_total_return
    })

# Main processing
summary_results = []
for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        process_instrument(os.path.join(input_folder, file_name), output_folder, summary_results)

summary_df = pd.DataFrame(summary_results)
summary_csv_path = os.path.join(output_folder, "summary_results.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSummary results saved to {summary_csv_path}")
