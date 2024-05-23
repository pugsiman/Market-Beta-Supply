This is a small predictive model built over an implementation of Welch's [Simply Better Market Betas](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3371240).

A populator script creates daily datasets for all current estimator values in NASDAQ listings (approximating for the overall supply in the market), which is then used by the model to track the supply of high beta stocks and how it changes through time.
Short periods following changes in supply lower (higher) tend to lead to more (less) volatile returns in highly risky (high beta) stocks ("junk stocks"), which can be helpful for use in optimization of hedging, or utilized in [dispersion trading](https://quantpedia.com/strategies/dispersion-trading), and more.

The model also includes a measure (which will be seperated later) called beta dispersion, a measure of market vulnarablity that can be used for market timing, as outlined by Kuntz [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3684268)

![image](https://github.com/pugsiman/market-beta-supply/assets/12158433/1aa1a3ad-cee1-4baf-a6ac-bd856305aeb8)
