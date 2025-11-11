import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA
import yfinance as yf
from datetime import datetime

# get assests and data

tickers = [
    'SPY',  # broad US equity (market)
    'QQQ',  # tech-heavy US equity
    'IWM',  # small-cap equities
    'XLF',  # financials sector ETF
    'XLE',  # energy sector ETF
    'TLT',  # long-term US Treasury bonds
    'IEF',  # intermediate-term Treasury bonds
    'LQD'   # investment-grade corporate bonds
]

start = '2020-01-01'
end = datetime.today().strftime('%Y-%m-%d')

prices = yf.download(tickers, start=start, end=end)['Close']
prices = prices.dropna()    #drop if any ticker is missing

# simple portfolio
weights= np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1]) 
portfolio_name = 'Equal-wight portfolio' 

# returns 
returns = prices.pct_change().dropna()

# standardize returns
X = (returns - returns.mean()) / returns.std(ddof=0)

# pca
pca_model = SKPCA()
pca_model.fit(X)

# variance
explained_var = pca_model.explained_variance_ratio_
cum_explained = np.cumsum(explained_var)

# components loadings - eigen vectors
loadings = pd.DataFrame(pca_model.components_.T, index=tickers,
                        columns=[f'PC{i+1}' for i in range(len(tickers))])

# factor time series(scores)
scores = pd.DataFrame(pca_model.transform(X), index=X.index,
                      columns=[f'PC{i+1}' for i in range(len(tickers))])

# portfolio returns
port_returns = (returns * weights).sum(axis=1)

# portfolio exposure to each PC = weights dot loadings for that PC
portfolio_exposure = loadings.multiply(weights, axis=0).sum(axis=0)

# remove 1st PC from returns
pc1 = pca_model.components_[0]

pc1_series = scores['PC1']      # time series of PC1

# contribution of PC! to each asset at each time = outer product of PC! and pc1_series
asset_contributution_pc1 = pd.DataFrame(np.outer(pc1_series.values, pc1), index=X.index, columns=tickers)

# PC1 contribution to portfolio = sum over assets (weights * asset_contri)
portfolio_contrib_pc1 = (asset_contributution_pc1 * weights).sum(axis=1)

#  neutralized portfolio return 
port_returns_neutral = port_returns - portfolio_contrib_pc1

# compare risk before and after neutralization
vol_before = port_returns.std() * np.sqrt(252)  #annualized std ~ risk
vol_after = port_returns_neutral.std() * np.sqrt(252)

print("Annualized volatility (before):", vol_before)
print("Annualized volatility (after removing PC1):", vol_after)
print(f"PC1 explains {explained_var[0]*100:.1f}% of variance (first PC).")
print("\nTop loadings for PC1 (assets most correlated with PC1):")
print(loadings['PC1'].abs().sort_values(ascending=False).head(6))


# plots

plt.figure(figsize=(8,4))
plt.plot(cum_explained[:10]*100, marker='o')
plt.title("Cumulativr variance explained")
plt.xlabel('Number of PC')
plt.ylabel('variance explained(%)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
loadings['PC1'].sort_values(ascending=True).plot(kind='bar')
plt.title("pc1 Loadings")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(port_returns.cumsum(), label='original')
plt.plot(port_returns_neutral.cumsum(), label='pc1 neutralized')
plt.legend()
plt.title("Cumulative Returns: Original vs Neutralized Portfolio")
plt.show()