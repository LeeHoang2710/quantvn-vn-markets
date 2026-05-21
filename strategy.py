import numpy as np
import pandas as pd


def _find_price_column(df: pd.DataFrame) -> str:
    for col in ("Close", "close"):
        if col in df.columns:
            return col
    raise ValueError("Input DataFrame must contain a 'Close' or 'close' column.")


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logic:
    - Use EMA(10) and EMA(30) to define short-term trend.
    - Use RSI(14) to avoid chasing overbought moves.
    - Buy signal (1): EMA10 > EMA30 and RSI crosses up through 55.
    - Sell signal (-1): EMA10 < EMA30 or RSI drops below 45.
    - No action (0): otherwise.

    Output:
    - signal: event-based trading signal in {1, 0, -1}
    - position: held position in {0, 1}
    """
    data = df.copy()
    price_col = _find_price_column(data)
    close = pd.to_numeric(data[price_col], errors="coerce")

    data["ema_fast"] = close.ewm(span=10, adjust=False).mean()
    data["ema_slow"] = close.ewm(span=30, adjust=False).mean()
    data["rsi_14"] = _rsi(close, period=14)

    trend_up = data["ema_fast"] > data["ema_slow"]
    trend_down = data["ema_fast"] < data["ema_slow"]

    rsi_cross_up = (data["rsi_14"].shift(1) <= 55) & (data["rsi_14"] > 55)
    rsi_weak = data["rsi_14"] < 45

    data["signal"] = 0
    data.loc[trend_up & rsi_cross_up, "signal"] = 1
    data.loc[trend_down | rsi_weak, "signal"] = -1

    # Convert event signals into a held long-only position for local backtests.
    position_state = []
    current_position = 0

    for signal in data["signal"].fillna(0).astype(int):
        if signal == 1:
            current_position = 1
        elif signal == -1:
            current_position = 0
        position_state.append(current_position)

    data["position"] = position_state
    return data
