# Predicting stocks using technical indicators (atr, lwma)

This trading strategy is designed for the [Quantiacs](https://quantiacs.com/contest) platform, which hosts competitions
for trading algorithms. Detailed information about the competitions is available on
the [official Quantiacs website](https://quantiacs.com/contest).

## How to Run the Strategy

### In an Online Environment

The strategy can be executed in an online environment using Jupiter or JupiterLab on
the [Quantiacs personal dashboard](https://quantiacs.com/personalpage/homepage). To do this, clone the template in your
personal account.

### In a Local Environment

To run the strategy locally, you need to install the [Quantiacs Toolbox](https://github.com/quantiacs/toolbox).

## Strategy Overview

This notebook provides a trading strategy for the NASDAQ-100 index using technical indicators to filter and weight
assets based on their volatility and liquidity. The core function (`strategy`) calculates the Average True Range (ATR)
and compares it with a threshold to filter assets. It then adjusts the weights of these assets using a Linear Weighted
Moving Average (LWMA) based on their money volume share. The process ensures only sufficiently liquid assets are traded.

Key components:

1. **ATR Filtering**: Uses ATR to filter out assets with high volatility.
2. **Weight Adjustment**: Adjusts asset weights based on their share of total money volume.
3. **Liquidity Check**: Ensures sufficient liquidity before executing trades.
4. **Performance Analysis**: Computes and visualizes performance metrics.
5. **Validation and Output**: Checks and writes the final weights for competition submission.

```python
import xarray as xr

import qnt.ta as qnta
import qnt.data as qndata
import qnt.output as qnout
import qnt.stats as qns


def strategy(data, wma, limit):
    vol = data.sel(field="vol")
    liq = data.sel(field="is_liquid")
    close = data.sel(field="close")
    high = data.sel(field="high")
    low = data.sel(field="low")

    atr = qnta.atr(high=high, low=low, close=close, ma=14)
    ratio = atr / close
    weights = xr.where(ratio > limit, 0, 1)

    money_vol = vol * liq * close
    total_money_vol = money_vol.sum(dim='asset')
    money_vol_share = money_vol / total_money_vol

    return qnta.lwma(money_vol_share, wma) * weights


data = qndata.stocks.load_ndx_data(min_date="2005-01-01")
weights_1 = strategy(data, wma=135, limit=0.0205)


def get_enough_bid_for(weights_):
    time_traded = weights_.time[abs(weights_).fillna(0).sum('asset') > 0]
    is_strategy_traded = len(time_traded)
    if is_strategy_traded:
        return xr.where(weights_.time < time_traded.min(), data.sel(field="is_liquid"), weights_)
    return weights_


weights_new = get_enough_bid_for(weights_1)
weights_new = weights_new.sel(time=slice("2006-01-01", None))

weights = qnout.clean(output=weights_new, data=data, kind="stocks_nasdaq100")


def print_statistic(data, weights_all):
    import qnt.stats as qnstats

    stats = qnstats.calc_stat(data, weights_all)
    display(stats.to_pandas().tail(5))
    # graph
    performance = stats.to_pandas()["equity"]
    import qnt.graph as qngraph

    qngraph.make_plot_filled(performance.index, performance, name="PnL (Equity)", type="log")


print_statistic(data, weights)
qnout.check(weights, data, "stocks_nasdaq100")
qnout.write(weights)  # To participate in the competition, save this code in a separate cell.

```
