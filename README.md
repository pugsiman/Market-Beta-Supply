This is a small predictive model built over an implementation for Welch Simply Better Market Betas (https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3371240).

A populator script creates daily datasets for all current estimator values in NASDAQ listings, which is then used by the model to track changes in the supply of high beta stocks, approximating for the total supply in the market.
Short periods following changes in supply lower (higher) tend to lead to more (less) volatile returns in highly risky and uncertain stocks ("junk stocks"), which can be helpful for optimizing current portfolio hedging or be utilized in [dispersion trading](https://quantpedia.com/strategies/dispersion-trading), for example.

![image](https://github.com/pugsiman/market-beta-supply/assets/12158433/1aa1a3ad-cee1-4baf-a6ac-bd856305aeb8)
