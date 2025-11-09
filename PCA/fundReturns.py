import numpy as np 
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime 
from sklearn.decomposition import PCA

# getting stock data 
tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 
           'NVDA', 'JPM', 'V', 'XOM', 'WMT', 'JNJ', 'PG', 
           'UNH', 'HD', 'MA', 'DIS', 'KO', 'PFE', 'PEP']

start = '2020-01-01'
end = datetime.today().strftime('%Y-%m-%d')
prices = yf.download(tickers, start=start, end=end)['Close']

# finding daily returns 
returns = prices.pct_change().dropna()

# standardization 
X = (returns - returns.mean()) / returns.std()

# applying pca
pca = PCA()
pca.fit(X)

# variance
explained_var = np.cumsum(pca.explained_variance_ratio_) * 100

# analysing top components 
num_components = 10
# eigenvector (loadings)
loadings = pd.DataFrame(pca.components_[:num_components],
                        columns = tickers,
                        index = [f'PC{i+1}' for i in range(num_components)])

# plot of variance 
plt.figure(figsize=(8,5))
plt.plot(explained_var, marker='o')
plt.title("Cumulative variance Explained by PCA Components")
plt.xlabel("Number of components")
plt.ylabel("variance explained (%)")
plt.grid(True)
plt.show()

print("Top 5 principal components loadings:")
print(loadings.round(3))