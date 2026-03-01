from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco

# Initialize the API
app = FastAPI(title="Enterprise Quantitative Portfolio Engine")

# Allow Next.js frontend to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected input from the frontend
class PortfolioRequest(BaseModel):
    tickers: List[str]
    start_date: str = "2023-01-01"
    end_date: str = "2026-01-01"
    initial_investment: float = 10000.0

# --- 1. Data Fetching & Math Engine ---
def fetch_market_data(tickers: List[str], start_date: str, end_date: str):
    try:
        all_tickers = tickers + ['SPY']
        prices_dict = {}
        
        for ticker in all_tickers:
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if ticker_data.empty:
                raise ValueError(f"No data found for {ticker}.")
                
            if 'Adj Close' in ticker_data.columns:
                prices_dict[ticker] = ticker_data['Adj Close'].squeeze()
            elif 'Close' in ticker_data.columns:
                prices_dict[ticker] = ticker_data['Close'].squeeze()
            else:
                raise ValueError(f"Pricing column not found for {ticker}.")
                
        prices = pd.DataFrame(prices_dict).dropna()
        daily_returns = prices.pct_change().dropna()
        
        basket_returns = daily_returns[tickers]
        benchmark_returns = daily_returns['SPY']
        
        cov_matrix_annual = basket_returns.cov() * 252
        expected_returns = basket_returns.mean() * 252
        
        return expected_returns, cov_matrix_annual, basket_returns, benchmark_returns
    except Exception as e:
        raise ValueError(f"Error fetching market data: {str(e)}")

def optimize_portfolio(expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    
    optimized_result = sco.minimize(
        portfolio_variance, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    optimal_weights = optimized_result.x
    expected_risk = np.sqrt(portfolio_variance(optimal_weights))
    expected_return = np.sum(expected_returns * optimal_weights)
    
    return optimal_weights, expected_return, expected_risk

# --- 2. Monte Carlo Stress Test Engine ---
def run_monte_carlo(expected_return, expected_risk, initial_investment, days=30, simulations=10000):
    daily_return = expected_return / 252
    daily_volatility = expected_risk / np.sqrt(252)
    
    drift = daily_return - (0.5 * daily_volatility**2)
    Z = np.random.normal(0, 1, (days, simulations))
    
    daily_simulated_returns = np.exp(drift + daily_volatility * Z)
    
    price_paths = np.zeros_like(daily_simulated_returns)
    price_paths[0] = initial_investment
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_simulated_returns[t]
        
    final_values = price_paths[-1]
    
    var_95 = initial_investment - np.percentile(final_values, 5)
    cvar_95 = initial_investment - final_values[final_values < np.percentile(final_values, 5)].mean()
    
    return float(var_95), float(cvar_95)

# --- 3. Historical Backtest Engine ---
def run_backtest(optimal_weights, basket_returns, benchmark_returns, initial_investment):
    portfolio_daily_returns = (basket_returns * optimal_weights).sum(axis=1)
    
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod() * initial_investment
    benchmark_cumulative = (1 + benchmark_returns).cumprod() * initial_investment
    
    history = []
    for date, port_val, bench_val in zip(portfolio_cumulative.index, portfolio_cumulative, benchmark_cumulative):
        history.append({
            "date": date.strftime('%Y-%m-%d'),
            "portfolio": round(float(port_val), 2),
            "benchmark": round(float(bench_val), 2)
        })
    
    rolling_max = portfolio_cumulative.cummax()
    drawdown = (portfolio_cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    risk_free_rate = 0.02
    annualized_return = portfolio_daily_returns.mean() * 252
    annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    return {
        "final_portfolio_value": round(float(portfolio_cumulative.iloc[-1]), 2),
        "final_benchmark_value": round(float(benchmark_cumulative.iloc[-1]), 2),
        "max_drawdown_percent": round(float(max_drawdown) * 100, 2),
        "sharpe_ratio": round(float(sharpe_ratio), 2),
        "outperformed_market": bool(portfolio_cumulative.iloc[-1] > benchmark_cumulative.iloc[-1]),
        "history": history
    }

# --- 4. The Master API Endpoint ---
@app.post("/api/optimize")
def build_and_test_portfolio(request: PortfolioRequest):
    try:
        exp_returns, cov_matrix, basket_ret, bench_ret = fetch_market_data(
            request.tickers, request.start_date, request.end_date
        )
        weights, opt_return, opt_risk = optimize_portfolio(exp_returns, cov_matrix)
        var_95, cvar_95 = run_monte_carlo(opt_return, opt_risk, request.initial_investment)
        backtest_results = run_backtest(weights, basket_ret, bench_ret, request.initial_investment)
        
        allocation = {
            ticker: round(float(w) * 100, 2) 
            for ticker, w in zip(request.tickers, weights)
        }
        
        return {
            "status": "success",
            "data": {
                "allocation": allocation,
                "future_projections": {
                    "expected_annual_return_percent": round(float(opt_return) * 100, 2),
                    "expected_annual_risk_percent": round(float(opt_risk) * 100, 2)
                },
                "stress_test_30_days": {
                    "value_at_risk_95": round(var_95, 2),
                    "conditional_value_at_risk_95": round(cvar_95, 2)
                },
                "historical_backtest": backtest_results
            }
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")