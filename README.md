This is a small predictive model built over an implementation of Welch's [Simply Better Market Betas](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3371240).

A populator script creates datasets for all current estimator values in NASDAQ listings (approximating for the overall supply in the market), which is then used by the model to track the time varying supply of high beta.

Short periods following changes in supply lower tend to lead to more volatile returns in highly risky stocks ("junk stocks"), which can be helpful in optimization of hedging, or utilized in [dispersion trading](https://quantpedia.com/strategies/dispersion-trading), and more. Moreover, under equilibrium principle, the supply of such stocks matches demand. In that view it is arguably a proxy of demand and positioning for risk in the market.

The model also includes a measure (which will be seperated later) called beta dispersion, a measure of market vulnarablity that can be used for market timing, as outlined by Kuntz [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3684268)

![image](https://github.com/pugsiman/market-beta-supply/assets/12158433/1aa1a3ad-cee1-4baf-a6ac-bd856305aeb8)
