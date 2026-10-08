import os
import json
import time
from datetime import datetime, timezone

try:
    import ccxt
except ImportError:  # pragma: no cover - handled gracefully at runtime
    ccxt = None

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BTC & ETH Trading Lab", page_icon="📈", layout="wide")


def require_ccxt():
    if ccxt is None:
        raise RuntimeError("ccxt is not installed. Install the app requirements before using market data or live trading.")

# Live orders are disabled by default and require an explicit per-order confirmation.
DEFAULTS = {
    "exchange_id": "binance",
    "symbol": "ETH/USDT",
    "timeframe": "1h",
    "candles": 500,
    "fast_ema": 12,
    "slow_ema": 26,
    "rsi_period": 14,
    "rsi_buy_min": 50,
    "rsi_buy_max": 70,
    "rsi_sell_min": 30,
    "rsi_sell_max": 50,
    "stop_loss_pct": 2.0,
    "risk_pct": 1.0,
    "starting_cash": 10000.0,
    "fee_pct": 0.1,
    "slippage_pct": 0.05,
    "poll_seconds": 60,
    "cooldown_candles": 3,
}
SIGNAL_FILE = "signals.csv"
TRADES_FILE = "backtest_trades.csv"
LIVE_STATE_FILE = "live_bot_state.json"
MAX_LIVE_ORDER_USDT = 100.0
MAX_DAILY_LOSS_USDT = 10.0

@st.cache_data(ttl=45, show_spinner=False)
def fetch_candles(exchange_id, symbol, timeframe, limit):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "timeout": 15000})
    if not exchange.has.get("fetchOHLCV"):
        raise RuntimeError(f"{exchange_id} does not support OHLCV candles via CCXT.")
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("The exchange returned no candles.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Exclude the currently-forming candle to avoid intrabar signals.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    return df.reset_index(drop=True)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.where(avg_loss != 0, 100)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50)
    return out

def add_indicators(df, cfg):
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=cfg["fast_ema"], adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=cfg["slow_ema"], adjust=False).mean()
    out["rsi"] = rsi(out["close"], cfg["rsi_period"])
    prev_fast = out["ema_fast"].shift(1)
    prev_slow = out["ema_slow"].shift(1)
    cross_up = (prev_fast <= prev_slow) & (out["ema_fast"] > out["ema_slow"])
    cross_down = (prev_fast >= prev_slow) & (out["ema_fast"] < out["ema_slow"])
    out["signal"] = "HOLD"
    buy = cross_up & out["rsi"].between(cfg["rsi_buy_min"], cfg["rsi_buy_max"], inclusive="both")
    sell = cross_down & out["rsi"].between(cfg["rsi_sell_min"], cfg["rsi_sell_max"], inclusive="both")
    out.loc[buy, "signal"] = "BUY"
    out.loc[sell, "signal"] = "SELL"
    return out

def append_signal(row, symbol, timeframe):
    record = {
        "detected_at_utc": datetime.now(timezone.utc).isoformat(),
        "candle_time_utc": row["timestamp"].isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": row["signal"],
        "close": float(row["close"]),
        "ema_fast": float(row["ema_fast"]),
        "ema_slow": float(row["ema_slow"]),
        "rsi": float(row["rsi"]),
    }
    frame = pd.DataFrame([record])
    frame.to_csv(SIGNAL_FILE, mode="a", header=not os.path.exists(SIGNAL_FILE), index=False)
    # Optional Telegram notification. Credentials are read from environment variables only.
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        try:
            import requests
            msg = (f"{symbol} signal: {record['signal']}\\n"
                   f"Close: {record['close']:.2f}\\nRSI: {record['rsi']:.1f}\\n"
                   f"Candle (UTC): {record['candle_time_utc']}\\n"
                   "Alert only — no order was placed.")
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg},
                timeout=8,
            ).raise_for_status()
        except Exception as exc:
            st.warning(f"Telegram notification failed: {exc}")

def load_live_state():
    if not os.path.exists(LIVE_STATE_FILE):
        return {"utc_date": datetime.now(timezone.utc).date().isoformat(),
                "daily_realized_pnl": 0.0, "position": None}
    with open(LIVE_STATE_FILE, "r", encoding="utf-8") as state_file:
        state = json.load(state_file)
    if not isinstance(state, dict) or "utc_date" not in state or "position" not in state:
        raise RuntimeError("Live state file is invalid; live orders are disabled until it is reviewed.")
    return state

def save_live_state(state):
    temp_file = LIVE_STATE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)
    os.replace(temp_file, LIVE_STATE_FILE)

def make_live_exchange():
    api_key = os.getenv("BINANCE_API_KEY", "").strip()
    secret = os.getenv("BINANCE_API_SECRET", "").strip()
    if not api_key or not secret:
        raise RuntimeError("Set BINANCE_API_KEY and BINANCE_API_SECRET in the app environment.")
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": secret,
        "enableRateLimit": True,
        "timeout": 15000,
        "options": {"defaultType": "spot"},
    })
    exchange.load_markets()
    return exchange

def place_live_stop(exchange, symbol, quantity, stop_price):
    amount = float(exchange.amount_to_precision(symbol, quantity))
    trigger = float(exchange.price_to_precision(symbol, stop_price))
    return exchange.create_order(
        symbol, "stop_market", "sell", amount, None, {"stopPrice": trigger}
    )

def record_live_exit(state, quantity, exit_price, fee_pct, timestamp_ms):
    position = state["position"]
    closed_quantity = min(float(quantity), float(position["quantity"]))
    fraction = closed_quantity / float(position["quantity"])
    event_date = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).date().isoformat()
    today = datetime.now(timezone.utc).date().isoformat()
    stale_day = state["utc_date"] != event_date
    if event_date == today and stale_day:
        state["utc_date"] = today
        state["daily_realized_pnl"] = 0.0
    reference_price = float(position["entry_price"] if stale_day else position.get("day_start_price", position["entry_price"]))
    fee = closed_quantity * float(exit_price) * fee_pct / 100
    if position.get("opened_date") == event_date:
        fee += float(position.get("entry_fee_usdt", 0.0)) * fraction
    pnl = closed_quantity * (float(exit_price) - reference_price) - fee
    if event_date == state["utc_date"]:
        state["daily_realized_pnl"] = float(state.get("daily_realized_pnl", 0.0)) + pnl
    remaining = float(position["quantity"]) - closed_quantity
    if remaining <= 1e-10:
        state["position"] = None
    else:
        position["quantity"] = remaining
        position["entry_fee_usdt"] = float(position.get("entry_fee_usdt", 0.0)) * (1 - fraction)
    return pnl

def reconcile_live_position(exchange, state, fee_pct):
    position = state.get("position")
    if not position:
        return
    if position.get("protection_error"):
        raise RuntimeError(position["protection_error"])
    stop_id = position.get("stop_order_id")
    if not stop_id:
        raise RuntimeError("The bot position has no recorded exchange stop; live orders are blocked.")
    order = exchange.fetch_order(stop_id, position["symbol"])
    status = order.get("status")
    if status == "closed":
        filled = float(order.get("filled") or 0)
        average = float(order.get("average") or 0)
        timestamp = int(order.get("lastTradeTimestamp") or order.get("timestamp") or time.time() * 1000)
        if filled <= 0 or average <= 0:
            raise RuntimeError("Stop order closed without usable fill details; review Binance before trading.")
        record_live_exit(state, filled, average, fee_pct, timestamp)
        if state.get("position"):
            state["position"]["protection_error"] = "Stop partially filled; remaining position requires manual review."
            save_live_state(state)
            raise RuntimeError(state["position"]["protection_error"])
        save_live_state(state)
    elif status not in ("open", "new"):
        position["protection_error"] = f"Exchange stop status is {status!r}; review Binance before trading."
        save_live_state(state)
        raise RuntimeError(position["protection_error"])

def roll_live_day(exchange, state):
    today = datetime.now(timezone.utc).date().isoformat()
    if state["utc_date"] == today:
        return
    position = state.get("position")
    if position:
        ticker = exchange.fetch_ticker(position["symbol"])
        mark = float(ticker.get("last") or 0)
        if mark <= 0:
            raise RuntimeError("Could not mark the open position; live orders are disabled.")
        position["day_start_price"] = mark
    state["utc_date"] = today
    state["daily_realized_pnl"] = 0.0
    save_live_state(state)

def live_daily_pnl(exchange, state):
    pnl = float(state.get("daily_realized_pnl", 0.0))
    position = state.get("position")
    if position:
        ticker = exchange.fetch_ticker(position["symbol"])
        mark = float(ticker.get("last") or 0)
        if mark <= 0:
            raise RuntimeError("Could not price the open bot position; live orders are disabled.")
        pnl += float(position["quantity"]) * (mark - float(position.get("day_start_price", position["entry_price"])))
        if position.get("opened_date") == state["utc_date"]:
            pnl -= float(position.get("entry_fee_usdt", 0.0))
    return pnl

def live_entry_quantity(exchange, symbol, price, daily_pnl, cfg):
    remaining_loss = min(MAX_DAILY_LOSS_USDT, MAX_DAILY_LOSS_USDT + daily_pnl)
    stop_fraction = cfg["stop_loss_pct"] / 100
    fee_fraction = cfg["fee_pct"] / 100
    loss_per_unit = price * (stop_fraction + fee_fraction * 2)
    if remaining_loss <= 0 or loss_per_unit <= 0:
        return 0.0
    quantity = min(MAX_LIVE_ORDER_USDT / price, remaining_loss / loss_per_unit)
    quantity = float(exchange.amount_to_precision(symbol, quantity))
    market = exchange.market(symbol)
    minimum = (market.get("limits", {}).get("amount", {}).get("min") or 0)
    minimum_cost = (market.get("limits", {}).get("cost", {}).get("min") or 0)
    if quantity < minimum or quantity * price < minimum_cost:
        return 0.0
    return quantity

def ensure_clean_market_account(exchange, symbol):
    base = symbol.split("/")[0]
    balance = exchange.fetch_balance()
    base_total = float((balance.get(base) or {}).get("total") or 0)
    market_minimum = exchange.market(symbol).get("limits", {}).get("amount", {}).get("min") or 0
    if base_total >= max(float(market_minimum), 1e-8):
        raise RuntimeError(f"The account already holds {base}; use a dedicated bot account or resolve the balance first.")
    if exchange.fetch_open_orders(symbol):
        raise RuntimeError("This market has open orders not tracked by the bot; resolve them before enabling entries.")

def render_live_controls(exchange_id, symbol, signal, cfg, enabled):
    if not enabled:
        return
    st.subheader("Live Binance spot controls")
    st.warning(
        "Each confirmed entry places a market buy and then a Binance stop-market sell. "
        "The $10 daily guard covers this app's tracked position only; gaps, fees, and "
        "slippage can exceed it. Keep API withdrawals disabled and use an IP restriction."
    )
    if exchange_id != "binance" or symbol not in ("BTC/USDT", "ETH/USDT"):
        st.error("Live controls are limited to Binance spot BTC/USDT and ETH/USDT.")
        return
    try:
        exchange = make_live_exchange()
        state = load_live_state()
        reconcile_live_position(exchange, state, cfg["fee_pct"])
        roll_live_day(exchange, state)
        mark = float(exchange.fetch_ticker(symbol).get("last") or 0)
        if mark <= 0:
            raise RuntimeError("Could not get a current market price.")
        daily_pnl = live_daily_pnl(exchange, state)
        save_live_state(state)
    except Exception as exc:
        st.error(f"Live controls are blocked: {exc}")
        return

    st.metric("Bot P/L today (estimated)", f"{daily_pnl:,.2f} USDT")
    position = state.get("position")
    if position:
        st.caption(
            f"Bot position: {position['symbol']} · {position['quantity']:.8g} · "
            f"entry {position['entry_price']:.4f} · stop {position['stop_price']:.4f}"
        )
        if position["symbol"] != symbol:
            st.info("Close the bot-managed position in the selected market before switching markets.")
            return
        if signal != "SELL":
            st.info("The position remains protected by its exchange stop. A SELL signal is required for a manual exit.")
            return
        with st.form("live_sell_confirmation"):
            confirmation = st.text_input(f"Type SELL {symbol} to confirm the market exit")
            submitted = st.form_submit_button("Confirm and exit position")
        if submitted:
            if confirmation.strip().upper() != f"SELL {symbol}":
                st.error("Confirmation did not match the required text.")
                return
            try:
                exchange.cancel_order(position["stop_order_id"], symbol)
                position["stop_order_id"] = None
                position["protection_error"] = "Exit in progress; refresh the page if interrupted."
                save_live_state(state)
                order = exchange.create_market_sell_order(symbol, float(position["quantity"]))
                filled = float(order.get("filled") or 0)
                average = float(order.get("average") or mark)
                if filled <= 0:
                    raise RuntimeError("Exit order returned no filled quantity.")
                timestamp = int(order.get("lastTradeTimestamp") or order.get("timestamp") or time.time() * 1000)
                record_live_exit(state, filled, average, cfg["fee_pct"], timestamp)
                if state.get("position"):
                    remaining = state["position"]["quantity"]
                    state["position"]["protection_error"] = "Exit partially filled; restoring protection."
                    save_live_state(state)
                    stop = place_live_stop(exchange, symbol, remaining, state["position"]["stop_price"])
                    state["position"]["stop_order_id"] = str(stop["id"])
                    state["position"]["protection_error"] = None
                save_live_state(state)
                st.success(f"Market exit filled: {filled:.8g} {symbol.split('/')[0]}.")
            except Exception as exc:
                if state.get("position") and not state["position"].get("stop_order_id"):
                    try:
                        stop = place_live_stop(
                            exchange, symbol, state["position"]["quantity"], state["position"]["stop_price"]
                        )
                        state["position"]["stop_order_id"] = str(stop["id"])
                        state["position"]["protection_error"] = None
                        save_live_state(state)
                    except Exception as stop_exc:
                        state["position"]["protection_error"] = (
                            f"CRITICAL: exit failed and stop could not be restored ({stop_exc}). "
                            "Review Binance immediately."
                        )
                        save_live_state(state)
                st.error(f"Exit failed or needs review: {exc}")
        return

    if signal != "BUY":
        st.info("No bot position is open. A BUY signal is required to preview an entry.")
        return
    try:
        ensure_clean_market_account(exchange, symbol)
    except Exception as exc:
        st.error(f"Live entry blocked: {exc}")
        return
    if daily_pnl <= -MAX_DAILY_LOSS_USDT:
        st.error("Daily bot loss limit reached; new entries are blocked until the next UTC day.")
        return
    quantity = live_entry_quantity(exchange, symbol, mark, daily_pnl, cfg)
    if quantity <= 0:
        st.error("The safe order size is below this market's minimum order size.")
        return
    notional = quantity * mark
    stop_price = mark * (1 - cfg["stop_loss_pct"] / 100)
    st.caption(
        f"Preview: buy about {notional:,.2f} USDT of {symbol}; stop-market near "
        f"{stop_price:,.4f} USDT. Position risk is sized against the remaining daily limit."
    )
    with st.form("live_buy_confirmation"):
        confirmation = st.text_input(f"Type BUY {symbol} to confirm the market entry and attached stop")
        submitted = st.form_submit_button("Confirm live entry")
    if not submitted:
        return
    if confirmation.strip().upper() != f"BUY {symbol}":
        st.error("Confirmation did not match the required text.")
        return

    buy_order = None
    try:
        current_mark = float(exchange.fetch_ticker(symbol).get("last") or 0)
        current_pnl = live_daily_pnl(exchange, state)
        quantity = live_entry_quantity(exchange, symbol, current_mark, current_pnl, cfg)
        if current_mark <= 0 or quantity <= 0 or current_pnl <= -MAX_DAILY_LOSS_USDT:
            raise RuntimeError("Price or daily loss limit changed; entry is no longer allowed.")
        quote_budget = min(MAX_LIVE_ORDER_USDT / (1 + cfg["fee_pct"] / 100), quantity * current_mark)
        buy_order = exchange.create_order(
            symbol, "market", "buy", None, None, {"quoteOrderQty": quote_budget}
        )
        filled = float(buy_order.get("filled") or 0)
        average = float(buy_order.get("average") or current_mark)
        if filled <= 0 or average <= 0:
            raise RuntimeError("Entry order returned no usable fill; check Binance order history before retrying.")
        stop_price = average * (1 - cfg["stop_loss_pct"] / 100)
        state["position"] = {
            "symbol": symbol,
            "quantity": filled,
            "entry_price": average,
            "entry_fee_usdt": filled * average * cfg["fee_pct"] / 100,
            "opened_date": state["utc_date"],
            "day_start_price": average,
            "stop_price": stop_price,
            "stop_order_id": None,
            "protection_error": "Stop placement pending; do not trade until refreshed.",
        }
        save_live_state(state)
        try:
            stop = place_live_stop(exchange, symbol, filled, stop_price)
            state["position"]["stop_order_id"] = str(stop["id"])
            state["position"]["protection_error"] = None
            save_live_state(state)
            st.success(f"Entry filled and exchange stop placed for {symbol}.")
        except Exception as stop_exc:
            try:
                emergency = exchange.create_market_sell_order(symbol, filled)
                emergency_filled = float(emergency.get("filled") or 0)
                emergency_average = float(emergency.get("average") or current_mark)
                if emergency_filled > 0:
                    timestamp = int(emergency.get("lastTradeTimestamp") or emergency.get("timestamp") or time.time() * 1000)
                    record_live_exit(state, emergency_filled, emergency_average, cfg["fee_pct"], timestamp)
                    if state.get("position"):
                        remaining = state["position"]["quantity"]
                        stop = place_live_stop(exchange, symbol, remaining, state["position"]["stop_price"])
                        state["position"]["stop_order_id"] = str(stop["id"])
                        state["position"]["protection_error"] = None
                    save_live_state(state)
                    st.error(f"Stop placement failed; emergency market close submitted. Stop error: {stop_exc}")
                else:
                    raise RuntimeError("Emergency close returned no fill.")
            except Exception as close_exc:
                state["position"]["protection_error"] = (
                    f"CRITICAL: stop placement and emergency close failed ({stop_exc}; {close_exc}). "
                    "Review Binance immediately."
                )
                save_live_state(state)
                st.error(state["position"]["protection_error"])
    except Exception as exc:
        st.error(f"Live entry failed; verify Binance order history before retrying: {exc}")

def run_backtest(df, cfg):
    # Long-only, one position at a time; signal at candle close, simulated fill next candle open.
    cash = float(cfg["starting_cash"])
    equity = cash
    position = None
    trades = []
    cooldown_until = -1
    fee = cfg["fee_pct"] / 100
    slip = cfg["slippage_pct"] / 100
    risk_fraction = cfg["risk_pct"] / 100
    stop_fraction = cfg["stop_loss_pct"] / 100
    warmup = max(cfg["slow_ema"], cfg["rsi_period"]) + 2

    for i in range(warmup, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        close = float(row["close"])
        next_open = float(nxt["open"])
        low = float(nxt["low"])
        high = float(nxt["high"])

        if position is not None:
            stop_price = position["entry"] * (1 - stop_fraction)
            # Conservative assumption: if a candle touches stop, fill at stop with slippage.
            if low <= stop_price:
                exit_price = stop_price * (1 - slip)
                gross = position["qty"] * (exit_price - position["entry"])
                exit_fee = position["qty"] * exit_price * fee
                pnl = gross - position["entry_fee"] - exit_fee
                cash += position["qty"] * exit_price - exit_fee
                trades.append({**position, "exit_time": nxt["timestamp"], "exit": exit_price,
                               "reason": "STOP_LOSS", "pnl": pnl, "return_pct": pnl / position["notional"] * 100})
                position = None
                cooldown_until = i + cfg["cooldown_candles"]
                continue
            if row["signal"] == "SELL":
                exit_price = next_open * (1 - slip)
                gross = position["qty"] * (exit_price - position["entry"])
                exit_fee = position["qty"] * exit_price * fee
                pnl = gross - position["entry_fee"] - exit_fee
                cash += position["qty"] * exit_price - exit_fee
                trades.append({**position, "exit_time": nxt["timestamp"], "exit": exit_price,
                               "reason": "SELL_SIGNAL", "pnl": pnl, "return_pct": pnl / position["notional"] * 100})
                position = None
                cooldown_until = i + cfg["cooldown_candles"]
                continue

        if position is None and row["signal"] == "BUY" and i >= cooldown_until:
            entry = next_open * (1 + slip)
            stop_distance = entry * stop_fraction
            risk_budget = max(0.0, cash * risk_fraction)
            qty_by_risk = risk_budget / stop_distance if stop_distance > 0 else 0
            # No leverage; position notional is capped at available cash.
            qty_by_cash = cash / (entry * (1 + fee)) if entry > 0 else 0
            qty = min(qty_by_risk, qty_by_cash)
            if qty > 0:
                notional = qty * entry
                entry_fee = notional * fee
                cash -= notional + entry_fee
                position = {"entry_time": nxt["timestamp"], "entry": entry, "qty": qty,
                            "notional": notional, "entry_fee": entry_fee}
    # Mark any open position at final close; include exit costs.
    if position is not None:
        exit_price = float(df.iloc[-1]["close"]) * (1 - slip)
        gross = position["qty"] * (exit_price - position["entry"])
        exit_fee = position["qty"] * exit_price * fee
        pnl = gross - position["entry_fee"] - exit_fee
        cash += position["qty"] * exit_price - exit_fee
        trades.append({**position, "exit_time": df.iloc[-1]["timestamp"], "exit": exit_price,
                       "reason": "END_OF_DATA", "pnl": pnl, "return_pct": pnl / position["notional"] * 100})
    trades_df = pd.DataFrame(trades)
    final_equity = cash
    if not trades_df.empty:
        final_equity = float(cfg["starting_cash"]) + trades_df["pnl"].sum()
    summary = {
        "Starting cash": float(cfg["starting_cash"]),
        "Ending equity": float(final_equity),
        "Net P/L": float(final_equity - cfg["starting_cash"]),
        "Return %": float((final_equity / cfg["starting_cash"] - 1) * 100),
        "Closed trades": int(len(trades_df)),
        "Win rate %": float((trades_df["pnl"] > 0).mean() * 100) if not trades_df.empty else 0.0,
        "Fees % per side": float(cfg["fee_pct"]),
        "Slippage % per side": float(cfg["slippage_pct"]),
    }
    return summary, trades_df

st.title("📈 BTC & ETH Trading Lab")
st.caption("BTC/USDT & ETH/USDT · Rule-based EMA + RSI · Alerts · Backtesting · Optional live controls")
st.warning("Research software, not financial advice. Crypto is volatile; backtests can be misleading. Live controls are off by default and require explicit confirmation for each entry or exit.")

with st.sidebar:
    st.header("Market")
    exchange_id = st.selectbox("Exchange", ["binance", "bybit", "okx", "kraken"], index=0)
    symbol = st.selectbox("Market symbol", ["ETH/USDT", "BTC/USDT"])
    timeframe = st.selectbox("Candle timeframe", ["5m", "15m", "30m", "1h", "4h", "1d"], index=3)
    candle_count = st.slider("Candles to load", 200, 1000, 500, 50)
    st.header("Strategy")
    fast_ema = st.number_input("Fast EMA", min_value=2, max_value=100, value=12)
    slow_ema = st.number_input("Slow EMA", min_value=3, max_value=200, value=26)
    rsi_period = st.number_input("RSI period", min_value=2, max_value=50, value=14)
    rsi_buy_min = st.slider("BUY RSI minimum", 1, 99, 50)
    rsi_buy_max = st.slider("BUY RSI maximum", 1, 99, 70)
    rsi_sell_min = st.slider("SELL RSI minimum", 1, 99, 30)
    rsi_sell_max = st.slider("SELL RSI maximum", 1, 99, 50)
    st.header("Risk & backtest")
    trial_funds_enabled = st.checkbox("Enable trial funds", value=False)
    trial_funds_usdt = st.number_input("Trial funds (USDT)", min_value=100.0, max_value=10000000.0, value=10000.0, step=500.0)
    starting_cash = st.number_input("Simulated starting cash (USDT)", min_value=100.0, max_value=10000000.0, value=10000.0, step=500.0)
    if trial_funds_enabled:
        st.caption("Using trial funds for paper trading mode. This value acts as the bankroll in the validation/backtest calculations.")
    risk_pct = st.slider("Risk budget per trade (%)", 0.1, 2.0, 1.0, 0.1)
    stop_loss_pct = st.slider("Stop-loss distance (%)", 0.5, 10.0, 2.0, 0.5)
    fee_pct = st.number_input("Estimated fee per side (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    slippage_pct = st.number_input("Estimated slippage per side (%)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    cooldown_candles = st.slider("Cooldown after exit (candles)", 0, 20, 3)
    live_enabled = st.checkbox("Enable live Binance spot controls", value=False)
    refresh = st.checkbox("Auto-refresh while page is open", value=False)
    poll_seconds = st.slider("Refresh interval (seconds)", 30, 300, 60, 30)

cfg = {
    "fast_ema": int(fast_ema), "slow_ema": int(slow_ema), "rsi_period": int(rsi_period),
    "rsi_buy_min": int(rsi_buy_min), "rsi_buy_max": int(rsi_buy_max),
    "rsi_sell_min": int(rsi_sell_min), "rsi_sell_max": int(rsi_sell_max),
    "stop_loss_pct": float(stop_loss_pct), "risk_pct": float(risk_pct),
    "starting_cash": float(trial_funds_usdt if trial_funds_enabled else starting_cash),
    "trial_funds_enabled": bool(trial_funds_enabled), "trial_funds_usdt": float(trial_funds_usdt),
    "fee_pct": float(fee_pct), "slippage_pct": float(slippage_pct), "cooldown_candles": int(cooldown_candles),
}
if cfg["fast_ema"] >= cfg["slow_ema"]:
    st.error("Fast EMA must be lower than slow EMA.")
    st.stop()
if cfg["rsi_buy_min"] > cfg["rsi_buy_max"] or cfg["rsi_sell_min"] > cfg["rsi_sell_max"]:
    st.error("Each RSI minimum must be less than or equal to its maximum.")
    st.stop()

try:
    with st.spinner("Fetching public market candles…"):
        raw = fetch_candles(exchange_id, symbol.strip().upper(), timeframe, int(candle_count))
    data = add_indicators(raw, cfg)
except Exception as exc:
    st.error(f"Could not load market data: {exc}")
    st.info("Check your internet connection, exchange availability, symbol format, and supported timeframe.")
    st.stop()

last = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else last
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last closed price", f"{last['close']:,.2f} USDT")
c2.metric("RSI", f"{last['rsi']:.1f}" if pd.notna(last["rsi"]) else "Warming up")
c3.metric("Fast EMA", f"{last['ema_fast']:,.2f}")
c4.metric("Latest signal", str(last["signal"]))

if last["signal"] in ("BUY", "SELL"):
    candle_key = f"{symbol}:{timeframe}:{last['timestamp'].isoformat()}:{last['signal']}"
    if st.session_state.get("last_alert_key") != candle_key:
        append_signal(last, symbol, timeframe)
        st.session_state["last_alert_key"] = candle_key
        st.toast(f"{last['signal']} signal logged", icon="🔔")
    st.subheader(f"🔔 {last['signal']} alert")
    st.write(f"Closed candle: {last['timestamp']} · Price: {last['close']:.2f} USDT · RSI: {last['rsi']:.1f}")
    st.caption("This is a strategy signal only. Confirm market conditions and your own risk limits before making any decision.")
else:
    st.info("No new signal on the latest closed candle. The system is monitoring for an EMA crossover confirmed by RSI.")

render_live_controls(exchange_id, symbol, str(last["signal"]), cfg, live_enabled)

left, right = st.columns([2, 1])
with left:
    st.subheader("Price and moving averages")
    chart = data.set_index("timestamp")[["close", "ema_fast", "ema_slow"]].rename(
        columns={"close": "Close", "ema_fast": f"EMA {fast_ema}", "ema_slow": f"EMA {slow_ema}"}
    )
    st.line_chart(chart)
with right:
    st.subheader("RSI")
    st.line_chart(data.set_index("timestamp")[["rsi"]])
    st.caption("Reference levels: 30 and 70. The strategy's thresholds are configurable in the sidebar.")

st.subheader("Recent closed candles")
show_cols = ["timestamp", "close", "ema_fast", "ema_slow", "rsi", "signal", "volume"]
st.dataframe(data[show_cols].tail(25).sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

st.subheader("Historical backtest")
st.caption("Simplified long-only simulation: signal on candle close, entry at next candle open, one position at a time, no leverage. Stops are approximated using candle lows; gaps, liquidity, and real execution may differ.")
summary, trades = run_backtest(data, cfg)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Net P/L", f"{summary['Net P/L']:,.2f} USDT")
m2.metric("Return", f"{summary['Return %']:.2f}%")
m3.metric("Closed trades", str(summary["Closed trades"]))
m4.metric("Win rate", f"{summary['Win rate %']:.1f}%")
m5.metric("Ending equity", f"{summary['Ending equity']:,.2f} USDT")
st.dataframe(pd.DataFrame([summary]), use_container_width=True, hide_index=True)
if not trades.empty:
    view = trades.copy()
    for col in ["entry_time", "exit_time"]:
        if col in view:
            view[col] = pd.to_datetime(view[col], utc=True).astype(str)
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Download backtest trades CSV", trades.to_csv(index=False).encode("utf-8"),
                       file_name=TRADES_FILE, mime="text/csv")
else:
    st.info("No trades were triggered in the loaded sample. Try a different timeframe or inspect the signal rules; do not tune solely to improve past results.")

st.subheader("Alert history")
if os.path.exists(SIGNAL_FILE):
    try:
        signals = pd.read_csv(SIGNAL_FILE)
        st.dataframe(signals.tail(100).iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("Download alert history CSV", signals.to_csv(index=False).encode("utf-8"),
                           file_name=SIGNAL_FILE, mime="text/csv")
    except Exception as exc:
        st.warning(f"Could not read signal history: {exc}")
else:
    st.caption("No alerts logged yet. Signals are recorded once per newly observed signal candle while the app is running.")

st.caption("Data provided by the selected exchange through CCXT. Public market data only; no exchange API keys are required. The app must remain open to monitor and send alerts.")
if refresh:
    st.caption(f"Auto-refresh is enabled: refresh the browser every {poll_seconds} seconds using your browser or deployment's refresh mechanism. This app does not run as a background service when closed.")
    # Streamlit rerun helper is optional and version-dependent.
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=poll_seconds * 1000, key="eth_autorefresh")
    except ImportError:
        st.warning("For auto-refresh, install streamlit-autorefresh (included in requirements.txt) and restart the app.")
