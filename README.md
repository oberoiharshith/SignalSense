# **SignalSense: Algorithmic Trading and Portfolio Management**  

## **Overview**  
SignalSense is a machine learning-driven algorithmic trading framework designed to optimize trading strategies across 29 financial instruments. It leverages **XGBoost models, feature engineering, and systematic backtesting** to enhance predictive accuracy and minimize risk.  

## **Key Highlights**  
- **Machine Learning for Trading** – Compared LSTM, Prophet, and XGBoost models, selecting XGBoost for its robustness in generalization.  
- **Risk Management** – Applied **rolling Sharpe ratio analysis, drawdown tracking, and stop-loss optimization** to mitigate losses.  
- **Backtesting & Forward Testing** – Evaluated trading strategies under historical and real-time market conditions to validate performance.  
- **Portfolio Diversification** – Conducted correlation analysis to optimize asset selection and improve risk-adjusted returns.  

## **Performance Summary**  

### **Backward Testing Results (Top 5 Instruments: May 2024 - Aug 2024)**  

| Instrument        | Best PnL  | Sharpe Ratio | Max Drawdown (%) | Total Return (%) |
|------------------|-----------|--------------|-------------------|------------------|
| LIGHT.CMDUSD     | 3470.378  | 0.338        | -0.019            | 0.347            |
| COPPER.CMDUSD    | 605.740   | 0.074        | -0.036            | 0.061            |
| SUGAR.CMDUSD     | 2776.105  | 0.280        | -0.037            | 0.278            |
| COTTON.CMDUSX    | 357.390   | 0.036        | -0.089            | 0.036            |
| USDJPY           | 383.533   | 0.086        | -0.017            | 0.038            |

### **Forward Testing Results (Top 5 Instruments: Sep 2024 - Dec 2024)**  

| Instrument        | PnL       | Sharpe Ratio | Max Drawdown (%) | Total Return (%) |
|------------------|-----------|--------------|-------------------|------------------|
| LIGHT.CMDUSD     | 277.640   | 0.113        | -0.023            | 0.028            |
| COPPER.CMDUSD    | 163.546   | 0.092        | -0.012            | 0.016            |
| SUGAR.CMDUSD     | 119.860   | 0.069        | -0.016            | 0.012            |
| COTTON.CMDUSX    | 103.810   | 0.069        | -0.010            | 0.010            |
| USDJPY           | 92.209    | 0.129        | -0.009            | 0.009            |

## **Key Observations**  
- Instruments like **LIGHT.CMDUSD and COPPER.CMDUSD** demonstrated consistent performance across backtesting and forward testing.  
- **USDJPY** showed sensitivity to market shifts, impacting forward test profitability.  
- Correlation analysis highlighted potential diversification benefits when combining select assets.  

## **Future Enhancements**  
✔️ Incorporate macroeconomic indicators and sentiment analysis for better predictions.  
✔️ Implement adaptive parameter tuning for real-time strategy adjustments.  
✔️ Expand to additional instruments and refine trading logic based on evolving market trends.  
