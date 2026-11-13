# CSE-Stock-Valuation-System

# AI-Enhanced Quantitative Stock Forecasting & Valuation System

**Deep Learning | Quantitative Finance | Time-Series Forecasting | Generative AI | Investment Analytics**

<img width="1181" height="650" alt="image" src="https://github.com/user-attachments/assets/295687d6-3410-4a73-ac88-45d625a6c98a" />
<img width="755" height="447" alt="image" src="https://github.com/user-attachments/assets/bab34ba7-883c-4f4a-aa15-60f1dee7ec78" />
<img width="761" height="376" alt="image" src="https://github.com/user-attachments/assets/cadef2d1-c627-4393-8f2d-6ed6c7589e14" />
<img width="781" height="623" alt="image" src="https://github.com/user-attachments/assets/694548b8-af19-41db-aae7-f9c800d6b84e" />
<img width="752" height="447" alt="image" src="https://github.com/user-attachments/assets/d1dec914-3c04-42be-a1c7-b40f80544ff9" />
<img width="1488" height="350" alt="image" src="https://github.com/user-attachments/assets/3ea8fb67-0878-4d67-b2d6-d84d07426cb8" />
<img width="675" height="547" alt="image" src="https://github.com/user-attachments/assets/fd456642-9592-4411-b361-8954e23de338" />

## Project Overview

Developed an AI-enhanced quantitative investment analytics system for the Colombo Stock Exchange (CSE), integrating **N-BEATS deep learning, financial valuation, technical indicators, strategy backtesting, and Ollama-powered Large Language Models (LLMs)**.

The system analyses voting and non-voting equities, forecasts stock prices, evaluates investment opportunities, and provides AI-assisted financial insights through a Flask-based dashboard.

## Key Features & Techniques

- **Deep Learning Forecasting:** N-BEATS neural architecture, time-series modelling, hyperparameter tuning, trend and seasonality analysis.
- **Forecast Evaluation:** MAE, RMSE, MAPE, R², chronological training/validation/testing.
- **Financial Modelling:** DCF-style intrinsic valuation, EPS, P/E ratio, NAVPS, growth projections and valuation-based mispricing analysis.
- **Technical Analysis:** RSI, MACD, EMA, Bollinger Bands, ATR, Fibonacci retracements and support/resistance levels.
- **Quantitative Backtesting:** Walk-forward strategy testing, Sharpe ratio, maximum drawdown, win rate and risk-adjusted performance.
- **Hybrid Investment Scoring:** Combines forecasting, fundamental valuation and technical signals to generate BUY, HOLD and SELL classifications.
- **Generative AI Integration:** Ollama-powered local LLM with prompt engineering, contextual financial data grounding and conversational stock analysis.
- **Interactive Application:** Flask-based dashboard for visualizing forecasts, valuations, trading signals and AI-generated financial interpretations.

## Model & Strategy Parameters

| Component | Configuration |
|---|---|
| Forecasting Model | N-BEATS (NeuralForecast / PyTorch) |
| Forecast Horizon | 60 Business Days |
| Validation / Test Horizons | 30 / 60 |
| Learning Rate | 0.0005 |
| Hyperparameter Configurations | 3 N-BEATS configurations |
| Loss Function | MAE |
| Technical Indicators | RSI(14), MACD(12,26,9), Bollinger(20,2), ATR(14) |
| Walk-Forward Backtesting | 504 Training / 126 Testing Bars |
| Risk Management | ATR Stop-Loss, Take-Profit, Trailing Stop |
| LLM Integration | Ollama REST API / Local LLM |
| Hybrid Scoring Weights | DCF 40% / Technical 35% / Forecast 25% |

## Technologies & Tools

**Programming:** Python, Pandas, NumPy, Scikit-learn

**Deep Learning:** N-BEATS, NeuralForecast, PyTorch

**Data Visualization:** Matplotlib

**Quantitative Finance:** DCF Valuation, Technical Analysis, Strategy Backtesting, Risk Analytics

**Generative AI:** Ollama, Large Language Models (LLMs), Prompt Engineering, Context-Aware Financial Analysis

**Application Development:** Flask, REST API, HTML/CSS

## Research Scope

**Market:** Colombo Stock Exchange (Sri Lanka)

**Selected Equities:** COMB, HNB, HNBF, LPL, RHL, SHOT

**Focus:** Voting vs. Non-Voting Share Price Behaviour, Forecast Accuracy, Intrinsic Valuation and Quantitative Investment Decision Support.

## Project Architecture

Historical Market & Financial Data  
→ Data Preprocessing  
→ N-BEATS Forecasting + DCF Valuation + Technical Analysis  
→ Walk-Forward Backtesting  
→ Hybrid Investment Scoring  
→ Flask Analytics Dashboard  
→ Ollama-Powered LLM Financial Assistant

## Implementation Notes

The current N-BEATS implementation uses historical price-series inputs. Additional exogenous-variable integration is part of the broader research architecture.

The valuation module implements an EPS-based DCF-style approach. The application uses a locally hosted Ollama LLM and requires the corresponding model service to run.

## Disclaimer

Developed for academic research and educational purposes. Forecasts and investment signals are experimental and do not constitute financial advice.

---

**Developed by Nirmalan Arjun**  
BSc (Hons) Data Science | 
