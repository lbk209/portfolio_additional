"""Price-path risk distributions from daily closing prices.

Four methods generate future closing-price paths under different assumptions.
Common results include threshold probabilities for terminal/min/max future
closes and whole-path extreme ranges. The known as-of close is excluded from
all future-close event observations.

Normal is a zero-expected-log-return, fixed-volatility *baseline*: compare the
same nonzero risk threshold across lookback-volatility scenarios, rather than
interpreting near-symmetric upside/downside probabilities as market evidence.
Its results are conditional on the model, not directional predictions. Other
methods can be compared against this deliberately simple reference, while
accounting for their differing drift assumptions and sampling procedures.

Threshold probabilities are model-conditional event frequencies, not disjoint
interval bins or guaranteed real-world probabilities. A shared threshold-event
backtest evaluates precisely the same event definition as the result table.
Historical windows may overlap and are statistically dependent.

Dependencies: numpy, pandas, matplotlib. FinanceDataReader is needed only
for load_prices() and can be installed with pip install finance-datareader.
Check splits, dividends, adjustments and data-feed conventions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence
import re
import scipy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler


TRADING_DAYS_PER_YEAR = 252

# These strings must match the corresponding PriceTargetAnalysis method names.
# When renaming/adding an analysis method, update this backtest allowlist too.
BACKTEST_METHODS = ("normal", "empirical", "bootstrap", "conditional_empirical")

_DEFAULT_MODEL = object()


def _price_types(price_type: str | Sequence[str]) -> tuple[tuple[str, ...], bool]:
    """Validate one price type or a sequence; report whether input was plural."""
    multiple = not isinstance(price_type, str)
    types = tuple(price_type) if multiple and isinstance(price_type, (list, tuple)) else (
        (price_type,) if isinstance(price_type, str) else ()
    )
    if not types or len(set(types)) != len(types) or any(
        not isinstance(p, str) or p not in ("terminal", "min", "max") for p in types
    ):
        raise ValueError("price_type must be 'terminal', 'min', 'max' or a nonempty list of these")
    return types, multiple


def _numeric_variants(values, *, arg_name: str, label, minimum: int) -> dict[str, int]:
    """Normalize integer, list/tuple, or named dict of integer configurations."""
    if isinstance(values, dict):
        if not values:
            raise ValueError(f"{arg_name} cannot be empty")
        pairs = list(values.items())
    elif isinstance(values, (int, np.integer)) and not isinstance(values, (bool, np.bool_)):
        pairs = [(label(int(values)), values)]
    elif isinstance(values, (list, tuple)):
        if not values:
            raise ValueError(f"{arg_name} cannot be empty")
        pairs = [(label(value), value) for value in values]
    else:
        raise TypeError(f"{arg_name} must be an integer, a list, or a dict")
    normalized = {}
    used = set()
    for name, value in pairs:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{arg_name} variant names must be nonempty strings")
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < minimum):
            raise ValueError(f"{arg_name} must contain integers >= {minimum}")
        value = int(value)
        if value in used:
            raise ValueError(f"{arg_name} contains a duplicate configuration: {value}")
        normalized[name] = value
        used.add(value)
    return normalized


def _validated_probabilities(probabilities: Sequence[float]) -> np.ndarray:
    """Check finite, unique quantile levels within [0, 1]."""
    if isinstance(probabilities, (bool, np.bool_)):
        raise ValueError("probabilities must be numeric, not boolean")
    try:
        arr = np.atleast_1d(np.asarray(probabilities, dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a sequence of finite numbers") from exc
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("probabilities must be a nonempty sequence of finite numbers")
    if np.any((arr < 0) | (arr > 1)) or len(np.unique(arr)) != len(arr):
        raise ValueError("probabilities must be unique numbers between 0 and 1")
    return arr


def _quantiles_style(frame: pd.DataFrame, *, pct: bool) -> Styler:
    """Format numeric quantile columns while preserving numeric DataFrame labels."""
    pattern = "{:+.1f}%" if pct else "{:,.2f}"
    styler = (frame.style
              .format(pattern, na_rep="—")
              .format_index(lambda p: f"p{100 * float(p):g}", axis="columns"))
    if np.isfinite(frame.to_numpy(dtype=float)).any():
        styler = styler.background_gradient(cmap="Blues", axis=None)
    return styler


def _recovery_style(frame: pd.DataFrame) -> Styler:
    """Format the recovery summary consistently for single/comparison tables."""
    styler = frame.style.format({
        "decline_probability_pct": "{:.1f}%",
        "recovery_probability_pct": "{:.1f}%",
        "median_recovery_days": "{:.1f}",
        "decline_pct": "{:.1f}%",
        "recovery_pct": "{:.1f}%",
    }, na_rep="—")
    cols = ["decline_probability_pct", "recovery_probability_pct"]
    if np.isfinite(frame[cols].to_numpy(dtype=float)).any():
        styler = styler.background_gradient(cmap="Blues", axis=None, subset=cols)
    return styler


def _validated_thresholds(thresholds: Sequence[float], *, pct: bool, mode: str) -> np.ndarray:
    """Validate the shared threshold event contract for tables and backtests."""
    if not isinstance(pct, (bool, np.bool_)):
        raise ValueError("pct must be a boolean")
    if mode not in ("above", "below", "mixed"):
        raise ValueError("mode must be 'above', 'below', or 'mixed'")
    if isinstance(thresholds, (bool, np.bool_)):
        raise ValueError("thresholds must be numeric, not boolean")
    try:
        arr = np.atleast_1d(np.asarray(thresholds, dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must be a sequence of finite numbers") from exc
    if arr.ndim != 1 or len(arr) == 0 or not np.isfinite(arr).all():
        raise ValueError("thresholds must be a nonempty sequence of finite numbers")
    if len(np.unique(arr)) != len(arr):
        raise ValueError("thresholds must not contain duplicates")
    if pct and np.any(arr <= -100):
        raise ValueError("percentage thresholds must be greater than -100%")
    if not pct and np.any(arr <= 0):
        raise ValueError("price thresholds must be positive")
    return arr


def _threshold_is_above(threshold: float, *, reference: float, pct: bool, mode: str) -> bool:
    """Mixed mode: below for negative returns, above for zero/positive returns."""
    if mode == "above":
        return True
    if mode == "below":
        return False
    return threshold >= (0.0 if pct else reference)


def _threshold_level(threshold: float, *, reference: float, pct: bool) -> float:
    return reference * (1.0 + threshold / 100.0) if pct else threshold


def _threshold_event(values: np.ndarray | float, *, level: float, above: bool) -> np.ndarray:
    """Consistent < / >= handling, including float-rounded exact boundaries.

    An index level of 110 must count as reaching +10% of 100 even if
    100 * (1 + .10) evaluates to 110.00000000000001 in binary arithmetic.
    Equality is recognized within a few machine ULPs; opposite events remain
    exact complements.
    """
    arr = np.asarray(values, dtype=float)
    boundary_equal = np.isclose(
        arr, level, rtol=8 * np.finfo(float).eps, atol=0.0,
    )
    if above:
        return (arr >= level) | boundary_equal
    return (arr < level) & ~boundary_equal




def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Validate daily closing prices; no forward filling or synthetic dates."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame with a Close column")
    if "Close" not in prices.columns:
        raise ValueError("prices must contain a 'Close' column")
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "Date"
    if frame.index.has_duplicates:
        raise ValueError("duplicated trading dates in prices")
    frame = frame.sort_index()
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    if frame.empty or frame["Close"].isna().any() or not np.isfinite(frame["Close"]).all():
        raise ValueError("Close contains missing/nonfinite values or is empty")
    if (frame["Close"] <= 0).any():
        raise ValueError("Close must be strictly positive")
    return frame


def load_prices(
    symbol: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    *,
    cache_dir: str | Path = "data",
    refresh: bool = False,
) -> pd.DataFrame:
    """Read FinanceDataReader daily data with an optional CSV cache.

    A cache is reused only if it reaches the requested end date (and starts
    within five calendar days after the requested start, allowing holidays).
    On a weekend/holiday, an end date newer than the last available session
    can trigger another download. Pass refresh=True to refresh current data.
    The requested end date is inclusive (request end+1 calendar day from FDR).
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("symbol must be a nonempty string")
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()
    if start_ts > end_ts:
        raise ValueError("start must be <= end")
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol)
    cache = folder / f"{filename}.csv"

    old = None
    if cache.exists():
        old = _clean_prices(pd.read_csv(cache, index_col="Date", parse_dates=["Date"]))
        # Trading calendar is unknown: allow weekend/holiday gaps up to 5 days.
        if (not refresh and old.index[0] <= start_ts + pd.Timedelta(days=5)
                and old.index[-1] >= end_ts):
            subset = old.loc[(old.index >= start_ts) & (old.index <= end_ts)]
            if not subset.empty:
                return subset
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:
        raise ImportError("Install FinanceDataReader: pip install finance-datareader") from exc
    fetch_start = min(start_ts, old.index[0]) if old is not None else start_ts
    # Download full range and merge to avoid gaps when older history is requested.
    raw = fdr.DataReader(
        symbol,
        fetch_start.strftime("%Y-%m-%d"),
        (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    new = _clean_prices(raw)
    if old is not None:
        new = pd.concat([old, new]).loc[lambda f: ~f.index.duplicated(keep="last")].sort_index()
        new = _clean_prices(new)
    new.to_csv(cache, index_label="Date")
    selected = new.loc[(new.index >= start_ts) & (new.index <= end_ts)]
    if selected.empty:
        raise ValueError(f"No prices for {symbol} in selected dates")
    return selected


def _validate_criterion_fields(indicator, window, match, tolerance, tolerance_mode):
    """Validate a price-state criterion without accessing market observations."""
    if not isinstance(indicator, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", indicator):
        raise ValueError("indicator must be a simple lowercase name")
    if (not isinstance(window, (int, np.integer))
            or isinstance(window, (bool, np.bool_)) or window < 2):
        raise ValueError("criterion window must be an integer >= 2")
    if match not in ("within", "above", "below"):
        raise ValueError("match must be 'within', 'above', or 'below'")
    if tolerance_mode not in ("relative", "absolute"):
        raise ValueError("tolerance_mode must be 'relative' or 'absolute'")
    if match == "within":
        if (tolerance is None or isinstance(tolerance, (bool, np.bool_))
                or not isinstance(tolerance, (int, float, np.integer, np.floating))
                or not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("match='within' requires a finite nonnegative tolerance")
    elif tolerance is not None:
        raise ValueError("tolerance must be omitted for match='above' or 'below'")


@dataclass(frozen=True)
class AsOfCriterion:
    """Match historical trailing indicators against their as-of values.

    Args:
        indicator: Indicator name; resolved as indicator_<name>(window) on
            PriceTargetAnalysis. Built-ins: volatility, drawdown, range_position.
        window: Rolling indicator lookback, in price observations (>= 2).
        tolerance: Nonnegative distance from the as-of indicator for
            match='within'; required only for that match mode.
        match: 'within' (inclusive two-sided range), 'above' (>= reference),
            or 'below' (<= reference). For above/below, omit tolerance.
        tolerance_mode: 'relative' (default: tolerance * abs(reference)) or
            'absolute' (tolerance in indicator units). Relevant to 'within'.

    Both historical and as-of indicator values use the same window. A zero
    reference cannot be used with relative tolerance for match='within'.
    """
    indicator: str
    window: int
    tolerance: Optional[float] = None
    match: str = "within"
    tolerance_mode: str = "relative"

    def __post_init__(self) -> None:
        _validate_criterion_fields(
            self.indicator, self.window, self.match,
            self.tolerance, self.tolerance_mode,
        )


@dataclass(frozen=True)
class FixedValueCriterion:
    """Match historical trailing indicators against an explicit fixed value.

    Args:
        indicator: Indicator name; resolved as indicator_<name>(window) on
            PriceTargetAnalysis. Built-ins: volatility, drawdown, range_position.
        window: Rolling indicator lookback, in price observations (>= 2).
        value: Fixed reference in indicator units, independent of as_of_date.
        tolerance: Nonnegative distance from value when match='within';
            required only for that match mode.
        match: 'within' (inclusive two-sided range), 'above' (>= value),
            or 'below' (<= value). For above/below, omit tolerance.
        tolerance_mode: 'relative' (default: tolerance * abs(value)) or
            'absolute' (tolerance in indicator units). Relevant to 'within'.

    A zero value cannot be used with relative tolerance for match='within'.
    Unlike AsOfCriterion, the comparison does not require a valid latest
    indicator value; only historical candidate values need to be finite.
    """
    indicator: str
    window: int
    value: float
    tolerance: Optional[float] = None
    match: str = "within"
    tolerance_mode: str = "relative"

    def __post_init__(self) -> None:
        _validate_criterion_fields(
            self.indicator, self.window, self.match,
            self.tolerance, self.tolerance_mode,
        )
        if (isinstance(self.value, (bool, np.bool_))
                or not isinstance(self.value, (int, float, np.integer, np.floating))
                or not np.isfinite(self.value)):
            raise ValueError("value must be a finite number")


Criterion = AsOfCriterion | FixedValueCriterion


@dataclass
class PathSample:
    """One model configuration's normalized H-step price trajectories.

    paths has shape (N, H) and excludes the starting close. Each historical
    path is rescaled to reference_price; it does NOT claim historical levels
    equal today's level. An empty path array indicates no usable paths.
    """
    name: str
    reference_price: float
    horizon_days: int
    paths: np.ndarray
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    start_dates: Optional[pd.DatetimeIndex] = None
    features: Optional[pd.DataFrame] = None

    @property
    def n_paths(self) -> int:
        return int(self.paths.shape[0])

    def _check(self) -> None:
        if self.status != "ok" or self.n_paths == 0:
            raise ValueError(f"{self.name}: {self.details.get('warning', 'no valid paths')}")





    def _path_values(
        self, price_type: str = "terminal", horizon_days: Optional[int] = None,
    ) -> np.ndarray:
        """One price per path: terminal, lowest, or highest *future* close.

        The known as-of close is deliberately excluded for all price types.
        All result tables use these same simulated/observed price paths; no
        resimulation or data download occurs when changing the horizon/type.
        """
        self._check()
        if price_type not in ("terminal", "min", "max"):
            raise ValueError("price_type must be 'terminal', 'min', or 'max'")
        h = self.horizon_days if horizon_days is None else horizon_days
        if (not isinstance(h, (int, np.integer))
                or isinstance(h, (bool, np.bool_))
                or not 1 <= h <= self.horizon_days):
            raise ValueError("horizon_days must be an integer between 1 and H")
        portion = self.paths[:, :int(h)]
        if price_type == "terminal":
            values = portion[:, -1]
        elif price_type == "min":
            values = portion.min(axis=1)
        else:
            values = portion.max(axis=1)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{self.name}: paths contain invalid future closing prices")
        return values

    def _distribution_prices(self, price_type: str) -> np.ndarray:
        """Compatibility helper for the existing histogram interface."""
        return self._path_values(price_type)

    def observed_paths(self, limit: int = 20) -> pd.DataFrame:
        """Inspect actual historical start dates (empirical and regime models)."""
        self._check()
        if self.start_dates is None:
            raise ValueError("This model has simulated, not dated historical paths")
        frame = pd.DataFrame({"start_date": self.start_dates,
                              "min_close": np.minimum(self.reference_price, self.paths.min(axis=1)),
                              "max_close": np.maximum(self.reference_price, self.paths.max(axis=1)),
                              "terminal_close": self.paths[:, -1]})
        if self.features is not None:
            frame = pd.concat([frame, self.features.reset_index(drop=True)], axis=1)
        return frame.head(limit)


@dataclass
class ModelResults:
    method: str
    variants: dict[str, PathSample]
    as_of_date: pd.Timestamp
    reference_price: float
    horizon_days: int
    historical_close: Optional[pd.Series] = None

    def summary(self, short=False) -> pd.DataFrame:
        record_short = {
            "as_of_date": str(self.as_of_date.date()),
            "reference_price": int(self.reference_price),
            "horizon_days": self.horizon_days
        }
        if short:
            return print(', '.join([f'{k} {v}' for k,v in record_short.items()]))
            
        records = []
        for key, item in self.variants.items():
            record = {"variant": key, "status": item.status, "n_paths": item.n_paths,
                      **record_short}
            record.update(item.details)
            records.append(record)
        return pd.DataFrame(records).set_index("variant")

    def plot_paths(
        self, *, pct=True, show_statistics=True, show_paths=False,
        max_paths=100, random_seed=42, history_days=None,
        width=None, height=None, show_method=False, fig=None,
        path_alpha=0.55, show_as_of_history=True,
    ):
        """Plot stored price paths with optional historical context.

        Statistics (median, pointwise P10-P90, daily min/max) use all
        future paths and are displayed only for t=0,...,H.
        Individual paths are sampled up to max_paths per variant.
        Their historical and future segments share one line and alpha.

        For paths with start_dates, preceding historical closes are shown
        when show_paths=True. The actual price history leading up to
        as_of_date can be displayed for every method, independently of
        show_paths. history_days=None uses horizon_days; 0 disables
        all historical prefixes.

        All historical prices are rebased to the relevant t=0 price.
        Legend entries show variant names unless show_method=True.
        An existing Figure receives additional traces. The Figure is
        returned without calling show().
        """
        import plotly.graph_objects as go
        from plotly.colors import hex_to_rgb, qualitative

        for name, flag in (
            ("pct", pct),
            ("show_statistics", show_statistics),
            ("show_paths", show_paths),
            ("show_method", show_method),
            ("show_as_of_history", show_as_of_history),
        ):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        if not (show_statistics or show_paths):
            raise ValueError("Enable show_statistics, show_paths, or both")

        for name, value, minimum in (
            ("max_paths", max_paths, 1),
            ("random_seed", random_seed, 0),
        ):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or value < minimum):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(path_alpha, (bool, np.bool_))
                or not isinstance(path_alpha, (int, float, np.integer, np.floating))
                or not np.isfinite(path_alpha) or not 0 < path_alpha <= 1):
            raise ValueError("path_alpha must be in (0, 1]")
        history_days = self.horizon_days if history_days is None else history_days
        if (isinstance(history_days, (bool, np.bool_))
                or not isinstance(history_days, (int, np.integer))
                or history_days < 0):
            raise ValueError("history_days must be None or a nonnegative integer")
        for name, dimension in (("width", width), ("height", height)):
            if dimension is not None and (
                isinstance(dimension, (bool, np.bool_))
                or not isinstance(dimension, (int, np.integer))
                or dimension <= 0
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        if fig is not None and not isinstance(fig, go.Figure):
            raise TypeError("fig must be a Plotly Figure or None")

        valid = {
            name: item for name, item in self.variants.items()
            if item.status == "ok" and item.n_paths > 0
        }
        skipped = {
            name: item for name, item in self.variants.items()
            if name not in valid
        }
        if skipped:
            _show_insufficient_data(self.method, skipped)
        if not valid:
            return fig

        if fig is None:
            fig = go.Figure()
            fig.add_hline(
                y=0 if pct else self.reference_price,
                line_color="gray", line_dash="dot", opacity=0.65,
            )
            fig.add_vline(x=0, line_color="gray", line_dash="dot", opacity=0.5)
            fig.update_layout(
                title=f"{self.method}: price paths",
                xaxis_title="Trading days from path start (t=0)",
                yaxis_title=(
                    "Return from path start (%)" if pct
                    else "Price / index level (rebased to as_of_date)"
                ),
                legend=dict(groupclick="togglegroup"),
                hovermode="closest",
                width=1200 if width is None else int(width),
                height=600 if height is None else int(height),
            )
        elif width is not None or height is not None:
            fig.update_layout(**{
                name: int(value)
                for name, value in (("width", width), ("height", height))
                if value is not None
            })

        reference = float(self.reference_price)
        future_x = np.arange(self.horizon_days + 1)
        rng = np.random.default_rng(random_seed)
        as_of_group = "__as_of_history__"
        used_groups = {
            trace.legendgroup for trace in fig.data
            if trace.legendgroup and trace.legendgroup != as_of_group
        }

        for variant, item in valid.items():
            if item.paths.shape != (item.n_paths, self.horizon_days):
                raise ValueError(f"{variant}: invalid paths shape")
            if not np.isfinite(item.paths).all() or (item.paths <= 0).any():
                raise ValueError(f"{variant}: invalid path prices")

            is_historical = item.start_dates is not None
            if is_historical and len(item.start_dates) != item.n_paths:
                raise ValueError(f"{variant}: start_dates count does not match paths")
            if show_paths and is_historical and history_days:
                if self.historical_close is None:
                    raise ValueError(
                        f"{variant}: historical_close is required for history_days > 0"
                    )
                if not isinstance(self.historical_close, pd.Series):
                    raise TypeError("historical_close must be a pandas Series")

            group = f"{self.method} / {variant}"
            color = qualitative.Plotly[len(used_groups) % len(qualitative.Plotly)]
            used_groups.add(group)
            r, g, b = hex_to_rgb(color)
            shown_name = group if show_method else variant

            selected = np.array([], dtype=int)
            if show_paths:
                selected = (
                    np.arange(item.n_paths)
                    if item.n_paths <= max_paths else
                    np.sort(rng.choice(item.n_paths, size=max_paths, replace=False))
                )

            caption = f"{shown_name} (n={item.n_paths:,}"
            if show_paths:
                caption += f", shown={len(selected):,}"
            caption += ")"

            for j, idx in enumerate(selected):
                prices = np.r_[reference, item.paths[idx]]
                x = future_x
                if is_historical:
                    date = pd.Timestamp(item.start_dates[idx])
                    date_info = f"Historical start: {date:%Y-%m-%d}"
                    if history_days > 0:
                        close = self.historical_close
                        pos = close.index.get_indexer([date])[0]
                        if pos < 0:
                            raise ValueError(
                                f"{variant}: historical start {date:%Y-%m-%d} "
                                "not found in historical_close"
                            )
                        begin = max(0, pos - int(history_days))
                        past = close.iloc[begin:pos].to_numpy(dtype=float)
                        start_close = float(close.iloc[pos])
                        if not np.isfinite(start_close) or start_close <= 0:
                            raise ValueError(f"{variant}: invalid historical start close")
                        prices = np.r_[reference * past / start_close, prices]
                        x = np.arange(-(pos - begin), self.horizon_days + 1)
                else:
                    date_info = f"Simulation path: {idx + 1}"

                returns = 100.0 * (prices / reference - 1.0)
                value_hover = (
                    "Return: %{y:.2f}%"
                    "<br>Normalized price: %{customdata:.2f}"
                    if pct else
                    "Normalized price: %{y:.2f}"
                    "<br>Return: %{customdata:.2f}%"
                )
                fig.add_trace(go.Scattergl(
                    x=x, y=returns if pct else prices,
                    customdata=prices if pct else returns,
                    mode="lines", name=caption,
                    legendgroup=group,
                    showlegend=(not show_statistics and j == 0),
                    line=dict(color=f"rgba({r},{g},{b},{path_alpha})", width=1),
                    hovertemplate=(
                        f"{shown_name}<br>{date_info}"
                        "<br>Trading day: %{x}<br>"
                        + value_hover + "<extra></extra>"
                    ),
                ))

            # All statistics start at t=0, regardless of historical prefixes.
            if show_statistics:
                p10, median, p90 = np.percentile(
                    item.paths, [10, 50, 90], axis=0
                )
                minimum = item.paths.min(axis=0)
                maximum = item.paths.max(axis=0)

                for label, values, dash, line_width in (
                    ("P10", p10, "solid", 0.8),
                    ("P90", p90, "solid", 0.8),
                    ("Min", minimum, "dot", 1.0),
                    ("Max", maximum, "dot", 1.0),
                    ("Median", median, "solid", 2.5),
                ):
                    prices = np.r_[reference, values]
                    y = (
                        100.0 * (prices / reference - 1.0)
                        if pct else prices
                    )
                    fig.add_trace(go.Scatter(
                        x=future_x, y=y, mode="lines",
                        name=caption if label == "Median" else f"{shown_name} {label}",
                        legendgroup=group,
                        showlegend=(label == "Median"),
                        line=dict(color=color, dash=dash, width=line_width),
                        fill="tonexty" if label == "P90" else None,
                        fillcolor=(
                            f"rgba({r},{g},{b},0.14)" if label == "P90" else None
                        ),
                        hovertemplate=(
                            f"{shown_name} — {label}"
                            "<br>Trading day: %{x}<br>"
                            + ("Return: %{y:.2f}%" if pct
                               else "Normalized price: %{y:.2f}")
                            + "<extra></extra>"
                        ),
                    ))

        # Actual price history preceding as_of_date; only one trace per Figure.
        already_shown = any(
            trace.legendgroup == as_of_group for trace in fig.data
        )
        if (show_as_of_history and history_days > 0 and not already_shown
                and self.historical_close is not None):
            if not isinstance(self.historical_close, pd.Series):
                raise TypeError("historical_close must be a pandas Series")
            actual = (
                self.historical_close.loc[:self.as_of_date]
                .iloc[-(int(history_days) + 1):]
            )
            if not actual.empty:
                actual_prices = actual.to_numpy(dtype=float)
                actual_returns = 100.0 * (actual_prices / reference - 1.0)
                actual_x = np.arange(1 - len(actual), 1)
                fig.add_trace(go.Scatter(
                    x=actual_x,
                    y=actual_returns if pct else actual_prices,
                    customdata=actual.index.strftime("%Y-%m-%d"),
                    mode="lines", name="As-of history",
                    legendgroup=as_of_group, showlegend=True,
                    line=dict(color="black", width=2.5),
                    hovertemplate=(
                        "As-of history<br>Date: %{customdata}"
                        "<br>Trading day: %{x}<br>"
                        + ("Return: %{y:.2f}%" if pct
                           else "Price / index: %{y:.2f}")
                        + "<extra></extra>"
                    ),
                ))

        return fig


    def threshold_probabilities(
        self, *, thresholds: float | Sequence[float], pct: bool = True,
        mode: str = "mixed", price_type: str | Sequence[str] = "terminal",
        horizon_days: Optional[int] = None, variant: Optional[str] = None,
        styled: bool = True, summary: bool = True,
    ) -> pd.DataFrame | Styler:
        """Threshold event frequencies for terminal/min/max future closes.

        thresholds accepts one number or a list. price_type accepts one
        string or a list of strings; a list produces a (variant, price_type)
        MultiIndex. The starting close is excluded from future events.
        """
        arr = _validated_thresholds(thresholds, pct=pct, mode=mode)
        if not isinstance(styled, (bool, np.bool_)):
            raise ValueError("styled must be a boolean")
        types, multiple = _price_types(price_type)
        if horizon_days is not None and (
            not isinstance(horizon_days, (int, np.integer))
            or isinstance(horizon_days, (bool, np.bool_))
            or not 1 <= horizon_days <= self.horizon_days
        ):
            raise ValueError("horizon_days must be an integer between 1 and H")
        if variant is not None and variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
        reference = float(self.reference_price)
        levels = [_threshold_level(float(t), reference=reference, pct=pct) for t in arr]
        above = [
            _threshold_is_above(float(t), reference=reference, pct=pct, mode=mode)
            for t in arr
        ]
        def formatted(t: float) -> str:
            if pct:
                value = f"{t:+g}%" if t != 0 else "0%"
            else:
                value = f"{t:,.15g}"
            return ("≥" if _threshold_is_above(t, reference=reference, pct=pct, mode=mode)
                    else "<") + value
        names = [formatted(float(t)) for t in arr]
        selected = ({variant: self.variants[variant]}
                    if variant is not None else self.variants)
        rows = {}
        for key, item in selected.items():
            for pt in types:
                row_key = (key, pt) if multiple else key
                if item.status != "ok" or item.n_paths == 0:
                    rows[row_key] = np.full(len(arr), np.nan)
                    continue
                values = item._path_values(pt, horizon_days=horizon_days)
                rows[row_key] = [
                    100.0 * np.mean(_threshold_event(values, level=level, above=is_above))
                    for level, is_above in zip(levels, above)
                ]
        result = pd.DataFrame.from_dict(rows, orient="index", columns=names)
        if multiple:
            result.index = pd.MultiIndex.from_tuples(
                result.index, names=["variant", "price_type"]
            )
        else:
            result.index.name = "variant"
        if summary:
            self.summary(short=True)
        if styled:
            return (result.style
                    .format("{:.1f}%", na_rep="—")
                    .background_gradient(cmap="Blues", vmin=0, vmax=100, axis=None))
        return result

    def return_quantiles(
        self, *, probabilities: float | Sequence[float], pct: bool = True,
        price_type: str | Sequence[str] = "terminal",
        horizon_days: Optional[int] = None, variant: Optional[str] = None,
        styled: bool = True, summary: bool = True,
    ) -> pd.DataFrame | Styler:
        """Quantiles of terminal/min/max future closes or arithmetic returns.

        probabilities accepts one number or a list. price_type accepts one
        string or a list of strings; the latter produces a
        (variant, price_type) MultiIndex. Quantiles are cumulative levels,
        not probabilities of exceeding the returned value.
        """
        levels = _validated_probabilities(probabilities)
        if not isinstance(pct, (bool, np.bool_)):
            raise ValueError("pct must be a boolean")
        if not isinstance(styled, (bool, np.bool_)):
            raise ValueError("styled must be a boolean")
        types, multiple = _price_types(price_type)
        if horizon_days is not None and (
            not isinstance(horizon_days, (int, np.integer))
            or isinstance(horizon_days, (bool, np.bool_))
            or not 1 <= horizon_days <= self.horizon_days
        ):
            raise ValueError("horizon_days must be an integer between 1 and H")
        if variant is not None and variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
        selected = ({variant: self.variants[variant]}
                    if variant is not None else self.variants)
        rows = {}
        for key, item in selected.items():
            for pt in types:
                row_key = (key, pt) if multiple else key
                if item.status != "ok" or item.n_paths == 0:
                    rows[row_key] = np.full(len(levels), np.nan)
                    continue
                values = item._path_values(pt, horizon_days=horizon_days)
                if pct:
                    values = 100.0 * (values / self.reference_price - 1.0)
                rows[row_key] = np.quantile(values, levels)
        frame = pd.DataFrame.from_dict(rows, orient="index", columns=levels)
        if multiple:
            frame.index = pd.MultiIndex.from_tuples(
                frame.index, names=["variant", "price_type"]
            )
        else:
            frame.index.name = "variant"
        if summary:
            self.summary(short=True)
        return _quantiles_style(frame, pct=pct) if styled else frame

    def recovery_after_decline(
        self, *, decline_pct: float = 5.0, recovery_pct: float = 0.0,
        horizon_days: Optional[int] = None, variant: Optional[str] = None,
        styled: bool = True, summary: bool = True,
    ) -> pd.DataFrame | Styler:
        """Frequency and timing of a recovery AFTER a daily-close decline.

        Decline: first FUTURE close <= reference*(1-decline_pct/100).
        Recovery: a LATER future close >= reference*(1+recovery_pct/100).
        recovery_pct may be negative but must exceed -decline_pct;
        e.g. decline_pct=20 and recovery_pct=-10 means recovery to -10%.
        Both thresholds are inclusive (within machine ULP rounding).
        Decline frequency uses all valid paths; recovery frequency uses only
        paths that first declined. Median recovery days uses only recovered
        paths, counting trading sessions from first decline to first recovery.
        Late declines have less time to recover within the specified horizon;
        these statistics are descriptive, not right-censoring-adjusted.
        """
        for name, value in (("decline_pct", decline_pct), ("recovery_pct", recovery_pct)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < decline_pct < 100:
            raise ValueError("decline_pct must be strictly between 0 and 100")
        if recovery_pct <= -decline_pct:
            raise ValueError("recovery_pct must be greater than -decline_pct")
        if not isinstance(styled, (bool, np.bool_)):
            raise ValueError("styled must be a boolean")
        h = self.horizon_days if horizon_days is None else horizon_days
        if (not isinstance(h, (int, np.integer))
                or isinstance(h, (bool, np.bool_)) or not 1 <= h <= self.horizon_days):
            raise ValueError("horizon_days must be an integer between 1 and H")
        h = int(h)
        if variant is not None and variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
        selected = (
            {variant: self.variants[variant]} if variant is not None
            else self.variants
        )
        drop_level = self.reference_price * (1.0 - float(decline_pct) / 100.0)
        restore_level = self.reference_price * (1.0 + float(recovery_pct) / 100.0)
        rows = {}
        for key, item in selected.items():
            row = {
                "horizon_days": h,
                "decline_pct": float(decline_pct),
                "recovery_pct": float(recovery_pct),
                "n_paths": item.n_paths,
                "decline_paths": 0,
                "recovered_paths": 0,
                "decline_probability_pct": np.nan,
                "recovery_probability_pct": np.nan,
                "median_recovery_days": np.nan,
                "status": item.status,
                "warning": item.details.get("warning") if item.status != "ok" else None,
            }
            if item.status == "ok" and item.n_paths > 0:
                paths = item.paths[:, :h]
                if not np.isfinite(paths).all() or (paths <= 0).any():
                    raise ValueError(f"{key}: paths contain invalid future closing prices")
                eps = 8 * np.finfo(float).eps
                dropped = ((paths <= drop_level)
                           | np.isclose(paths, drop_level, rtol=eps, atol=0.0))
                has_drop = dropped.any(axis=1)
                first_drop = dropped.argmax(axis=1)
                step = np.arange(h)[None, :]
                restored = (
                    ((paths >= restore_level)
                     | np.isclose(paths, restore_level, rtol=eps, atol=0.0))
                    & (step > first_drop[:, None])
                    & has_drop[:, None]
                )
                has_recovery = restored.any(axis=1)
                n_drop = int(has_drop.sum())
                n_recovery = int(has_recovery.sum())
                recovery_days = (
                    restored.argmax(axis=1)[has_recovery] - first_drop[has_recovery]
                )
                row.update(
                    decline_paths=n_drop,
                    recovered_paths=n_recovery,
                    decline_probability_pct=100.0 * n_drop / item.n_paths,
                    recovery_probability_pct=(100.0 * n_recovery / n_drop
                                              if n_drop else np.nan),
                    median_recovery_days=(float(np.median(recovery_days))
                                          if n_recovery else np.nan),
                )
            rows[key] = row
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "variant"
        if summary:
            self.summary(short=True)
        return _recovery_style(frame) if styled else frame

    def extreme_range(
        self, *, probability: float = 0.90,
        horizons: Optional[Sequence[int]] = None,
        variant: Optional[str] = None, summary=True
    ) -> pd.DataFrame:
        """Ranges containing at least the requested fraction of whole future paths.

        For each horizon, use the lower tail of future path minima and upper
        tail of future path maxima, allocating (1-probability)/2 to each side.
        Conservative order statistics keep the *observed* joint path coverage
        at least probability. The two bounds are not claimed to form the
        narrowest possible interval. This is model/sample coverage, not a
        guarantee about an actual future market path. Unlike pointwise
        quantile bands in plot_paths(), the bounds constrain whole paths
        through each specified horizon; coverage is tested separately at
        each horizon.

        Compare bounds across volatility lookbacks at fixed horizon/coverage;
        unlike terminal threshold_probabilities(), these are simultaneous
        boundaries for ALL future daily closes through that horizon. Under
        zero-log-drift Normal, price bounds need not be equally far above and
        below the reference close: exponentiating symmetric log movements
        produces asymmetric percentage price changes, not a bullish signal.
        """
        if (not isinstance(probability, (int, float, np.integer, np.floating))
                or isinstance(probability, (bool, np.bool_))
                or not np.isfinite(probability) or not 0 < probability < 1):
            raise ValueError("probability must be finite and strictly between 0 and 1")
        hs = (self.horizon_days,) if horizons is None else (
            (horizons,) if isinstance(horizons, (int, np.integer)) else tuple(horizons)
        )
        if (not hs or any(not isinstance(h, (int, np.integer))
                          or isinstance(h, (bool, np.bool_))
                          or h < 1 or h > self.horizon_days for h in hs)):
            raise ValueError("horizons must contain trading-day integers between 1 and H")
        hs = tuple(sorted(set(int(h) for h in hs)))
        if variant is not None and variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
        selected = ({variant: self.variants[variant]}
                    if variant is not None else self.variants)
        records = []
        for key, item in selected.items():
            for h in hs:
                row = dict(variant=key, horizon_days=h, probability=float(probability),
                           lower_price=np.nan, upper_price=np.nan,
                           empirical_coverage=np.nan, n_paths=item.n_paths,
                           status=item.status,
                           warning=item.details.get("warning") if item.status != "ok" else None)
                if item.status == "ok" and item.n_paths > 0:
                    lower_values = item._path_values("min", horizon_days=h)
                    upper_values = item._path_values("max", horizon_days=h)
                    n = item.n_paths
                    # At most floor((1-p)*N/2) paths may fall outside EACH
                    # bound, yielding joint coverage >= p (union bound).
                    k = int(np.floor((1.0 - probability) * n / 2.0 + 1e-10))
                    k = min(k, n - 1)
                    lower = float(np.sort(lower_values)[k])
                    upper = float(np.sort(upper_values)[n - 1 - k])
                    coverage = float(np.mean((lower_values >= lower) & (upper_values <= upper)))
                    row.update(lower_price=lower, upper_price=upper,
                               empirical_coverage=coverage)
                records.append(row)

        if summary:
            self.summary(short=True)
            
        return pd.DataFrame(records, columns=[
            "variant", "horizon_days", "probability", "lower_price", "upper_price",
            "empirical_coverage", "n_paths", "status", "warning",
        ])



    def plot_extreme_range(
        self, *, probability: float = 0.90,
        variant: Optional[str] = None,
    ) -> None:
        """Plot whole-path price envelopes over the full horizon.

        At each horizon, the bounds contain at least the requested fraction
        of complete sampled paths, unlike pointwise quantiles in plot_paths().
        They do not represent individual trajectories; coverage is evaluated
        separately at each horizon.

        variant=None plots all valid variants.
        Variant names appear only in the legend.
        Uses extreme_range() without new simulations.
        """
        if variant is not None and variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")

        selected = (
            self.variants if variant is None
            else {variant: self.variants[variant]}
        )
        valid = {
            name: item for name, item in selected.items()
            if item.status == "ok" and item.n_paths > 0
        }
        skipped = {
            name: item for name, item in selected.items()
            if name not in valid
        }

        if skipped:
            _show_insufficient_data(self.method, skipped)
        if not valid:
            return

        horizons = range(1, self.horizon_days + 1)
        fig, ax = plt.subplots(figsize=(10, 4.5))

        for name in valid:
            frame = self.extreme_range(
                probability=probability,
                horizons=horizons,
                variant=name, summary=False
            ).sort_values("horizon_days")

            x = frame["horizon_days"].to_numpy(dtype=int)
            low = frame["lower_price"].to_numpy(dtype=float)
            high = frame["upper_price"].to_numpy(dtype=float)

            upper_line, = ax.plot(
                x, high, linewidth=1.6, label=name,
            )
            color = upper_line.get_color()

            ax.plot(
                x, low, color=color,
                linestyle="--", linewidth=1.6,
            )
            ax.fill_between(
                x, low, high, color=color,
                alpha=0.12 if len(valid) > 1 else 0.18,
            )

        ax.axhline(
            self.reference_price,
            color="dimgray",
            linestyle=":",
            linewidth=1,
            alpha=0.8,
            label="Reference close",
        )

        ax.set(
            title=(
                f"{self.method}: {probability:.0%} requested "
                f"path envelope ({self.horizon_days} trading days)"
            ),
            xlabel="Horizon (trading days)",
            ylabel="Price / index level",
        )
        ax.grid(alpha=0.2)
        ax.legend(title="Solid: upper / Dashed: lower", loc='upper left')
        ax.xaxis.get_major_locator().set_params(integer=True)

        fig.tight_layout()
        plt.show()
        plt.close(fig)



    def price_distribution(
        self, *, price_type: str = "terminal", bin_width_pct: float = 5.0,
        variant: Optional[str] = None,
    ) -> pd.DataFrame:
        """Non-overlapping closing-price return bins for each model variant.

        price_type selects the future H-th close (terminal), or the lowest/highest
        of future daily closes (min/max). The known as-of close is excluded from
        all three observations. Bin boundaries are multiples of bin_width_pct
        percentage points, anchored to 0% return from reference_price:
        [lower_pct, upper_pct). Each variant's probabilities sum to 1.
        No resimulation or extra data download is performed.

        Return probability as a fraction and probability_pct as a percentage;
        price bounds are actual levels derived from reference_price.
        """
        if price_type not in ("terminal", "min", "max"):
            raise ValueError("price_type must be 'terminal', 'min', or 'max'")
        if not isinstance(bin_width_pct, (int, float, np.integer, np.floating)) or not np.isfinite(bin_width_pct) or bin_width_pct <= 0:
            raise ValueError("bin_width_pct must be a positive finite percentage-point width")
        width = float(bin_width_pct)
        if variant is not None and variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
        selected = ({variant: self.variants[variant]} if variant is not None
                    else self.variants)
        rows = []
        for key, item in selected.items():
            if item.status != "ok" or item.n_paths == 0:
                rows.append({"variant": key, "status": item.status, "price_type": price_type,
                             "bin_width_pct": width, "return_lower_pct": np.nan,
                             "return_upper_pct": np.nan, "price_lower": np.nan,
                             "price_upper": np.nan, "path_count": 0,
                             "n_paths": item.n_paths, "probability": np.nan,
                             "probability_pct": np.nan,
                             "warning": item.details.get("warning", "No valid paths")})
                continue
            values = item._distribution_prices(price_type)
            returns_pct = 100.0 * (values / self.reference_price - 1.0)
            # Integer bin indices avoid ambiguity from floating-point histogram
            # edges; every price belongs to exactly one half-open interval.
            bins = np.floor(returns_pct / width).astype(np.int64)
            first, last = int(bins.min()), int(bins.max())
            n_bins = last - first + 1
            if n_bins > 1000:
                raise ValueError(
                    f"{key}: {n_bins} price bins would be generated; "
                    "increase bin_width_pct to keep the output readable"
                )
            counts = np.bincount(bins - first, minlength=n_bins)
            for offset, count in enumerate(counts):
                low_pct = (first + offset) * width
                high_pct = low_pct + width
                rows.append({
                    "variant": key, "status": item.status, "price_type": price_type,
                    "bin_width_pct": width, "return_lower_pct": low_pct,
                    "return_upper_pct": high_pct,
                    "price_lower": self.reference_price * (1.0 + low_pct / 100.0),
                    "price_upper": self.reference_price * (1.0 + high_pct / 100.0),
                    "path_count": int(count), "n_paths": item.n_paths,
                    "probability": float(count / item.n_paths),
                    "probability_pct": float(100.0 * count / item.n_paths),
                    "warning": None,
                })
        columns = ["variant", "status", "price_type", "bin_width_pct",
                   "return_lower_pct", "return_upper_pct", "price_lower",
                   "price_upper", "path_count", "n_paths", "probability",
                   "probability_pct", "warning"]
        return pd.DataFrame(rows, columns=columns)



    def plot_price_distribution(
        self, *, price_type: str = "terminal", bin_width_pct: float = 5.0,
        variant: Optional[str] = None, pct: bool = True,
        x_min: Optional[float] = None, x_max: Optional[float] = None,
        kind: str = "hist", kde_bw_method: str | float = "scott",
    ) -> None:
        """Plot return distributions as a histogram or KDE density curve.
    
        kind='hist' (default): probability per return bin, in percent.
        kind='kde': Gaussian kernel density estimate using SciPy.
    
        pct=True uses percentage returns on the x-axis; pct=False uses prices.
        x_min/x_max clip the display without renormalizing probabilities.
        bin_width_pct applies only to hist; kde_bw_method applies only to KDE.
    
        KDE smoothness depends on bandwidth and sample composition. Historical
        paths may overlap, and simulated paths are not independent historical
        episodes. KDE does not increase the amount of independent evidence.
        Small samples or few distinct regimes can produce misleading shapes.
        Gaussian KDE can assign density below the -100% return boundary;
        such impossible values are not plotted, and density is not renormalized.
        """
        if kind not in ("hist", "kde"):
            raise ValueError("kind must be 'hist' or 'kde'")
        if not isinstance(pct, (bool, np.bool_)):
            raise ValueError("pct must be a boolean")
        if price_type not in ("terminal", "min", "max"):
            raise ValueError("Invalid price_type")
        if variant is not None and variant not in self.variants:
            raise ValueError(f"Unknown variant: {variant}")
    
        for name, bound in (("x_min", x_min), ("x_max", x_max)):
            if bound is not None and (
                not isinstance(bound, (int, float, np.integer, np.floating))
                or not np.isfinite(bound)
            ):
                raise ValueError(f"{name} must be a finite number or None")
    
        if not pct and x_min is not None and x_min < 0:
            raise ValueError("x_min cannot be negative in price mode")
    
        selected = (
            {variant: self.variants[variant]} if variant is not None
            else self.variants
        )
        valid = {
            key: item for key, item in selected.items()
            if item.status == "ok" and item.n_paths > 0
        }
        skipped = {key: item for key, item in selected.items() if key not in valid}
    
        if skipped:
            _show_insufficient_data(self.method, skipped)
        if not valid:
            return
    
        reference = float(self.reference_price)
    
        def to_axis(value):
            value = np.asarray(value)
            return value if pct else reference * (1.0 + value / 100.0)
    
        def to_pct(value):
            value = np.asarray(value)
            return value if pct else 100.0 * (value / reference - 1.0)
    
        # KDE uses the original path observations, not histogram bin counts.
        kde_models = {}
        if kind == "kde":
            try:
                from scipy.stats import gaussian_kde
            except ImportError as exc:
                raise ImportError(
                    "KDE requires SciPy: pip install scipy"
                ) from exc
    
            if isinstance(kde_bw_method, str):
                if kde_bw_method not in ("scott", "silverman"):
                    raise ValueError("Unknown KDE bandwidth method")
            elif (
                not isinstance(kde_bw_method, (int, float))
                or not np.isfinite(kde_bw_method)
                or kde_bw_method <= 0
            ):
                raise ValueError("kde_bw_method must be positive")
    
            for key, item in valid.items():
                values = item._distribution_prices(price_type)
                returns_pct = 100.0 * (values / reference - 1.0)
    
                scale = max(1.0, float(np.max(np.abs(returns_pct))))
                if (
                    len(returns_pct) < 2
                    or np.ptp(returns_pct) <= 1e-12 * scale
                ):
                    print(f"{key}: KDE skipped — insufficient variation.")
                    continue
    
                try:
                    kde = gaussian_kde(
                        returns_pct, bw_method=kde_bw_method
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    print(f"{key}: KDE skipped — {exc}")
                    continue
    
                kde_models[key] = kde
    
            valid = {
                key: item for key, item in valid.items()
                if key in kde_models
            }
            if not valid:
                print(f"{self.method}: No valid KDE distributions.")
                return
    
        # Determine a common horizontal range across all valid variants.
        lows, highs = [], []
        for key, item in valid.items():
            values = item._distribution_prices(price_type)
            returns_pct = 100.0 * (values / reference - 1.0)
            lows.append(float(np.quantile(returns_pct, 0.01)))
            highs.append(float(np.quantile(returns_pct, 0.99)))
    
        width_pct = float(bin_width_pct)
    
        if kind == "hist":
            # Preserve the original histogram range and bin alignment.
            default_lo = np.floor(min(lows) / width_pct - 1) * width_pct
            default_hi = np.ceil(max(highs) / width_pct + 1) * width_pct
        else:
            # KDE does not depend on histogram bin_width_pct.
            padding = max(0.1 * (max(highs) - min(lows)), 1e-6)
            default_lo = min(lows) - padding
            default_hi = max(highs) + padding
    
        left = float(x_min) if x_min is not None else float(to_axis(default_lo))
        right = float(x_max) if x_max is not None else float(to_axis(default_hi))
    
        if not pct:
            left = max(left, 0.0)
        if left >= right:
            raise ValueError("x_min must be less than x_max")
    
        fig, ax = plt.subplots(figsize=(10, 4.5))
    
        if kind == "hist":
            table = self.price_distribution(
                price_type=price_type,
                bin_width_pct=bin_width_pct,
                variant=variant,
            )
            labels = list(valid)
            bar_width_pct = width_pct / len(labels)
    
            for i, key in enumerate(labels):
                subset = table.loc[table["variant"] == key]
                lower_pct = subset["return_lower_pct"].to_numpy(dtype=float)
    
                # Split each common bin into adjacent variant bars.
                left_pct = lower_pct + i * bar_width_pct
                bar_width = (
                    bar_width_pct if pct
                    else reference * bar_width_pct / 100.0
                )
                ax.bar(
                    to_axis(left_pct),
                    subset["probability_pct"].to_numpy(dtype=float),
                    width=bar_width, align="edge", linewidth=0,
                    label=key, alpha=0.85,
                )
    
            # Preserve histogram bin-boundary ticks and guide lines.
            first = int(np.ceil(float(to_pct(left)) / width_pct - 1e-10))
            last = int(np.floor(float(to_pct(right)) / width_pct + 1e-10))
    
            if first <= last:
                n_boundaries = last - first + 1
                grid_step = (
                    1 if n_boundaries <= 65
                    else int(np.ceil(n_boundaries / 65))
                )
                for k in range(first, last + 1, grid_step):
                    ax.axvline(
                        float(to_axis(k * width_pct)),
                        color="gray", linewidth=0.45,
                        alpha=0.16, zorder=0,
                    )
    
                tick_step = max(1, int(np.ceil(n_boundaries / 11)))
                first_major = ((first + tick_step - 1) // tick_step) * tick_step
                tick_ids = list(range(first_major, last + 1, tick_step))
                if first <= 0 <= last and 0 not in tick_ids:
                    tick_ids.append(0)
                tick_ids.sort()
    
                ticks = [float(to_axis(k * width_pct)) for k in tick_ids]
                ax.set_xticks(ticks)
    
                if pct:
                    tick_labels = [f"{k * width_pct:g}%" for k in tick_ids]
                else:
                    tick_labels = [
                        f"{reference:,.2f}" if k == 0
                        else f"{tick:,.2f}" if abs(tick) < 100
                        else f"{tick:,.0f}"
                        for k, tick in zip(tick_ids, ticks)
                    ]
    
                ax.set_xticklabels(tick_labels, fontsize=9)
                for k, label in zip(tick_ids, ax.get_xticklabels()):
                    if k == 0:
                        label.set_fontweight("bold")
                        label.set_color("dimgray")
    
            ylabel = f"Probability per {width_pct:g}%-point bin (%)"
    
        else:
            # Shared evaluation grid, with no values below -100% return.
            x_grid = np.linspace(left, right, 600)
            return_grid = to_pct(x_grid)
            mask = return_grid > -100.0
    
            for key, item in valid.items():
                density = kde_models[key](return_grid[mask])
    
                # Change density units when x-axis is actual price.
                if not pct:
                    density = density * 100.0 / reference
    
                label = f"{key} (n={item.n_paths}"
                label += ")"
    
                ax.plot(
                    x_grid[mask], density,
                    linewidth=1.8, label=label,
                )
    
            ylabel = (
                "Probability density (per percentage point)" if pct
                else "Probability density (per price unit)"
            )
    
        ax.set_xlim(left, right)
    
        reference_x = 0.0 if pct else reference
        ax.axvline(
            reference_x, linewidth=1, linestyle="--",
            color="dimgray", alpha=0.8,
        )
        ax.text(
            0.99, 0.98, f"Reference close: {reference:,.2f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=9,
        )
    
        ax.set(
            title=f"{self.method}: {price_type} daily-close distribution ({kind})",
            xlabel=("Change from reference close (%)" if pct
                    else "Price / index level"),
            ylabel=ylabel,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(title="Variant")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


@dataclass
class BacktestResult:
    """Out-of-sample evaluation of a specified terminal/min/max threshold event."""
    method: str
    threshold: float
    pct: bool
    mode: str
    price_type: str
    horizon_days: int
    observations: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        frame = self.observations.dropna(
            subset=["predicted_probability", "realized_event"]
        )
        return pd.DataFrame([{
            "method": self.method,
            "threshold": self.threshold,
            "pct": self.pct,
            "mode": self.mode,
            "price_type": self.price_type,
            "horizon_days": self.horizon_days,
            "valid_forecasts": len(frame),
            "skipped_forecasts": len(self.observations) - len(frame),
            "mean_predicted_probability": frame["predicted_probability"].mean() if len(frame) else np.nan,
            "realized_event_rate": frame["realized_event"].mean() if len(frame) else np.nan,
            "brier_score": np.mean((frame["predicted_probability"] - frame["realized_event"])**2)
                           if len(frame) else np.nan,
        }])

    def plot_calibration(self, bins: int = 5) -> None:
        """Descriptive event calibration; overlapping forecast windows are dependent."""
        if bins < 2:
            raise ValueError("bins must be >=2")
        df = self.observations.dropna(
            subset=["predicted_probability", "realized_event"]
        ).copy()
        if df.empty:
            print(f"{self.method}: Insufficient data — no valid backtest forecasts.")
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        df["bin"] = pd.cut(
            df["predicted_probability"], bins=np.linspace(0, 1, bins + 1),
            include_lowest=True,
        )
        cal = df.groupby("bin", observed=True).agg(
            predicted=("predicted_probability", "mean"),
            observed=("realized_event", "mean"), n=("realized_event", "size"),
        )
        ax.scatter(cal["predicted"], cal["observed"],
                   s=20 + 5 * cal["n"], label="Backtest bins")
        ax.set(
            xlabel="Mean predicted event probability",
            ylabel="Observed event frequency",
            title=f"{self.method} threshold-event calibration (overlapping horizons)",
            xlim=(0, 1), ylim=(0, 1),
        )
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        plt.show()
        plt.close(fig)



def _show_insufficient_data(method: str, variants: dict[str, PathSample]) -> None:
    """Print unavailable variants without creating an empty figure."""
    if not variants:
        print(f"{method}: Insufficient data — no variants available.")
    else:
        for key, item in variants.items():
            print(f"{method} [{key}]: {item.status} — "
                  f"{item.details.get('warning', 'No valid price paths')}.")


class PriceTargetAnalysis:
    """One as-of date and horizon; no target price is fixed on construction.

    New regime indicators may be implemented as ``indicator_<name>(window)``
    returning a date-indexed pandas Series of finite or NaN values. Registering
    such methods requires no changes to conditional_empirical(). For example:

    def indicator_momentum(self, window):
        return self.close.pct_change(window)

    and AsOfCriterion('momentum', 20, 0.20).
    """

    def __init__(self, prices: pd.DataFrame, *, horizon_days: int,
                 as_of_date: Optional[str] = None):
        if not isinstance(horizon_days, (int, np.integer)) or horizon_days < 1:
            raise ValueError("horizon_days must be a positive integer")
        self.horizon_days = int(horizon_days)
        all_prices = _clean_prices(prices)
        # None means the latest observation in the *provided* frame, not a
        # live market lookup. The reference date is fixed on construction.
        requested = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else None
        self.prices = (all_prices.loc[all_prices.index <= requested].copy()
                       if requested is not None else all_prices.copy())
        if self.prices.empty:
            raise ValueError("No price available on or before as_of_date")
        self.as_of_date = self.prices.index[-1]
        self.close = self.prices["Close"]
        self.reference_price = float(self.close.iloc[-1])
        self.log_returns = np.log(self.close / self.close.shift(1))
        self._indicator_cache: dict[tuple[str, int], pd.Series] = {}


    def info(self) -> None:
        info = {
            "as_of_date": str(self.as_of_date.date()),
            "reference_price": int(self.reference_price),
            "horizon_days": self.horizon_days,
            "available_start": str(self.prices.index[0].date()),
            "available_closes": len(self.close),
        }
        _ = [print(f"{k}: {v}") for k,v in info.items()]


    def _empty(self, name: str, warning: str, **details) -> PathSample:
        return PathSample(name, self.reference_price, self.horizon_days,
                          np.empty((0, self.horizon_days)),
                          details={"warning": warning, **details}, status="insufficient_data")

    def _results(self, method: str, variants: dict[str, PathSample]) -> ModelResults:
        return ModelResults(method, variants, self.as_of_date, self.reference_price,
                            self.horizon_days, historical_close=self.close)

    def normal(
        self, *, lookback_days: int | Sequence[int] | dict[str, int] = (20, 60, 120),
        simulations: int = 20_000, random_seed: int = 42,
    ) -> ModelResults:
        """Zero-log-drift IID Gaussian baseline; lookback_days defines variants.

        Accept an integer, a list of integers, or a name-to-integer dict.
        All variants reuse the same random shocks. They represent volatility
        lookback sensitivity, not independent market forecasts.
        """
        _validate_simulations(simulations)
        specs = _numeric_variants(
            lookback_days, arg_name="lookback_days", label=lambda v: f"{v}d_vol", minimum=2
        )
        variants = {}
        available = self.log_returns.dropna()
        normal_shocks = np.random.default_rng(random_seed).standard_normal(
            size=(simulations, self.horizon_days)
        )
        for key, lb in specs.items():
            if len(available) < lb:
                variants[key] = self._empty(
                    key, f"Need {lb} returns; only {len(available)} available",
                    lookback_days=lb,
                )
                continue
            sigma_daily = float(available.iloc[-lb:].std(ddof=1))
            increments = sigma_daily * normal_shocks
            paths = self.reference_price * np.exp(np.cumsum(increments, axis=1))
            variants[key] = PathSample(
                key, self.reference_price, self.horizon_days, paths,
                details={
                    "lookback_days": lb,
                    "annualized_volatility": sigma_daily * np.sqrt(TRADING_DAYS_PER_YEAR),
                    "annual_log_drift": 0.0,
                    "simulations": simulations,
                },
            )
        return self._results("Normal", variants)

    def _history_date(self, history_start: Optional[str]) -> pd.Timestamp:
        """None means the earliest close in the as-of-truncated input data."""
        return self.close.index[0] if history_start is None else pd.Timestamp(history_start)

    def _historical_windows(
        self, *, history_start: Optional[str], sampling_step: int,
        min_samples: int,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
        """Complete historical paths starting at or after the effective date."""
        if (not isinstance(sampling_step, (int, np.integer))
                or isinstance(sampling_step, (bool, np.bool_)) or sampling_step < 1
                or min_samples < 1):
            raise ValueError("sampling_step and min_samples must be positive integers")
        close = self.close.to_numpy(dtype=float)
        dates = self.close.index
        start_date = self._history_date(history_start)
        latest_start = len(close) - self.horizon_days - 1
        start_index = int(dates.searchsorted(start_date))
        candidate = np.arange(start_index, latest_start + 1, sampling_step, dtype=int)
        details = {
            "history_start": str(start_date.date()),
            "available_paths": len(candidate),
            "min_samples": min_samples,
            "sampling_step": int(sampling_step),
            "first_start": str(dates[candidate[0]].date()) if len(candidate) else None,
            "last_start": str(dates[candidate[-1]].date()) if len(candidate) else None,
        }
        if len(candidate) == 0:
            return np.empty((0, self.horizon_days)), pd.DatetimeIndex([]), details
        offsets = np.arange(1, self.horizon_days + 1)
        paths = self.reference_price * close[candidate[:, None] + offsets] / close[candidate, None]
        return paths, dates[candidate], details

    def empirical(
        self, *, history_start: Optional[str] = None,
        sampling_step: int | Sequence[int] | dict[str, int] = 1,
        min_samples: int = 30, min_history_days: int = 252,
    ) -> ModelResults:
        """Observed normalized historical paths; sampling_step defines variants.

        Integer -> one named variant; list -> multiple automatically named
        variants; dict -> multiple custom-named variants. Sampling steps are
        observation counts, not calendar days.
        """
        if min_history_days < 2:
            raise ValueError("min_history_days must be >=2")
        specs = _numeric_variants(
            sampling_step, arg_name="sampling_step", label=lambda v: f"step_{v}", minimum=1
        )
        start_date = self._history_date(history_start)
        available_days = int((self.close.index >= start_date).sum())
        variants = {}
        for key, step in specs.items():
            paths, dates, details = self._historical_windows(
                history_start=history_start, sampling_step=step, min_samples=min_samples
            )
            details.update({
                "available_history_days": available_days,
                "min_history_days": min_history_days,
            })
            if available_days < min_history_days:
                variants[key] = self._empty(
                    key, f"Need {min_history_days} days of history, got {available_days}", **details
                )
            elif len(paths) < min_samples:
                variants[key] = self._empty(
                    key, f"Need {min_samples} historical paths, got {len(paths)}", **details
                )
            else:
                variants[key] = PathSample(
                    key, self.reference_price, self.horizon_days,
                    paths, details=details, start_dates=dates,
                )
        return self._results("Empirical", variants)

    def bootstrap(
        self, *, history_start: Optional[str] = None,
        block_sizes: int | Sequence[int] | dict[str, int] = (1, 5, 10, 20),
        simulations: int = 20_000, random_seed: int = 42,
        drift_adjustment: str = "none", min_history_returns: int = 252,
    ) -> ModelResults:
        """Moving-block bootstrap; block_sizes defines variants.

        Accept an integer, a list of integers, or a name-to-integer dict.
        block_size=1 is IID bootstrap. With 'zero', subtract the mean log
        return; 'none' retains the sample mean. Partial last block is trimmed.
        """
        _validate_simulations(simulations)
        if min_history_returns < 2:
            raise ValueError("min_history_returns must be >=2")
        if drift_adjustment not in ("none", "zero"):
            raise ValueError("drift_adjustment must be 'none' or 'zero'")
        specs = _numeric_variants(
            block_sizes, arg_name="block_sizes", label=lambda v: f"block_{v}", minimum=1
        )
        start_date = self._history_date(history_start)
        history = self.log_returns.loc[self.log_returns.index >= start_date].dropna()
        arr = history.to_numpy(dtype=float)
        if drift_adjustment == "zero" and len(arr):
            arr = arr - arr.mean()
        variants = {}
        for key, bs in specs.items():
            details = {
                "block_size": bs,
                "history_returns": len(arr),
                "history_start": str(start_date.date()),
                "first_return_date": str(history.index[0].date()) if len(history) else None,
                "last_return_date": str(history.index[-1].date()) if len(history) else None,
                "drift_adjustment": drift_adjustment,
                "simulations": simulations,
            }
            if len(arr) < max(min_history_returns, bs):
                variants[key] = self._empty(
                    key, f"Need at least {max(min_history_returns, bs)} returns; got {len(arr)}", **details
                )
                continue
            rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), bs]))
            block_count = (self.horizon_days + bs - 1) // bs
            starts = rng.integers(0, len(arr) - bs + 1, size=(simulations, block_count))
            indices = (starts[:, :, None] + np.arange(bs)).reshape(
                simulations, -1
            )[:, :self.horizon_days]
            increments = arr[indices]
            paths = self.reference_price * np.exp(np.cumsum(increments, axis=1))
            variants[key] = PathSample(
                key, self.reference_price, self.horizon_days, paths, details=details,
            )
        return self._results("Bootstrap", variants)

    def indicator_volatility(self, window: int) -> pd.Series:
        """Annualized sample stdev of the latest `window` daily log returns."""
        return self.log_returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)

    def indicator_drawdown(self, window: int) -> pd.Series:
        """Close relative to the highest close in the preceding window (incl. t)."""
        return self.close / self.close.rolling(window, min_periods=window).max() - 1

    def indicator_range_position(self, window: int) -> pd.Series:
        """Close's position (0..1) in the trailing window's closing-price range.

        Include the current close, use no future values, and return NaN when
        the trailing high equals its low (position is then undefined).
        """
        lo = self.close.rolling(window, min_periods=window).min()
        hi = self.close.rolling(window, min_periods=window).max()
        return ((self.close - lo) / (hi - lo)).where(hi > lo)

    def _indicator(self, criterion: Criterion) -> pd.Series:
        key = criterion.indicator, criterion.window
        if key not in self._indicator_cache:
            method = getattr(self, f"indicator_{criterion.indicator}", None)
            if method is None or not callable(method):
                raise ValueError(f"No indicator_{criterion.indicator}(window) method")
            series = method(criterion.window)
            if not isinstance(series, pd.Series) or not series.index.equals(self.close.index):
                raise ValueError("indicator must return a Series indexed exactly like close")
            self._indicator_cache[key] = pd.to_numeric(series, errors="coerce")
        return self._indicator_cache[key]


    def conditional_empirical(
        self, *,
        history_start: Optional[str] = None,
        criteria: Optional[
            Criterion | Sequence[Criterion] |
            dict[str, Criterion | Sequence[Criterion]]
        ] = None,
        sampling_step: int = 1,
    ) -> ModelResults:
        """Build historical forward paths matching trailing price-state criteria.

        Args:
            history_start:
                Earliest eligible historical path-start date (inclusive).
                None starts with the first available close. Earlier prices,
                when available, can still inform trailing indicator windows.
            criteria:
                One AsOfCriterion or FixedValueCriterion, a list/tuple of
                ANDed criteria (one 'matched' variant), or a dictionary
                mapping variant names to individual criteria/ANDed groups.
                None defaults to as-of volatility(window=20, tolerance=0.25)
                AND drawdown(window=126, tolerance=0.50); both use relative
                tolerance. Each variant is evaluated independently.
            sampling_step:
                Positive trading-observation interval between CANDIDATE
                start dates, starting at the first trading observation on
                or after history_start. Default 1 evaluates every date.
                Candidates are sampled BEFORE criteria matching, so larger
                steps can miss brief matching periods. This reduces directly
                overlapping starts but does not guarantee independence.

        Returns:
            ModelResults containing each variant's matched historical paths,
            start_dates, indicator features and criteria metadata. Every
            matching candidate is retained, including when sample counts are
            small; zero matches yield status='no_matches' and empty paths.

        Notes:
            Path starts require a complete horizon_days forward window of
            daily closing prices (the start close itself is excluded).
            Each path is rescaled to the analysis reference_price.
            history_start is also the anchor for sampling_step > 1; changing
            it may change which dates are considered even in shared periods.
            Paths may share future closes and are statistically dependent.
            No minimum sample count or episode count is imposed.
        """
        if criteria is None:
            groups = {
                "matched": (
                    AsOfCriterion("volatility", 20, 0.25),
                    AsOfCriterion("drawdown", 126, 0.50),
                )
            }
        elif isinstance(criteria, dict):
            groups = criteria
        else:
            groups = {"matched": criteria}
        if not groups:
            raise ValueError("At least one criterion variant is required")
        normalized = {}
        for variant_name, spec in groups.items():
            if not isinstance(variant_name, str) or not variant_name.strip():
                raise ValueError("Criterion variant names must be nonempty strings")
            if isinstance(spec, (AsOfCriterion, FixedValueCriterion)):
                group = (spec,)
            elif isinstance(spec, (list, tuple)):
                group = tuple(spec)
            else:
                raise TypeError(f"{variant_name}: expected Criterion or nonempty list")
            if not group:
                raise ValueError(f"{variant_name}: criteria cannot be empty")
            if not all(isinstance(c, (AsOfCriterion, FixedValueCriterion)) for c in group):
                raise TypeError(f"{variant_name}: criteria must contain Criterion objects")
            normalized[variant_name] = group

        if (not isinstance(sampling_step, (int, np.integer))
                or isinstance(sampling_step, (bool, np.bool_))
                or sampling_step < 1):
            raise ValueError("sampling_step must be a positive integer")

        n = len(self.close)
        latest_start = n - self.horizon_days - 1
        start_date = self._history_date(history_start)
        first = int(self.close.index.searchsorted(start_date))
        candidates = np.arange(first, latest_start + 1, sampling_step, dtype=int)
        candidate_index = self.close.index[candidates]
        common_details = {
            "history_start": str(start_date.date()),
            "sampling_step": int(sampling_step),
            "candidate_count": len(candidates),
            "available_history_days": int((self.close.index >= start_date).sum()),
            "first_candidate_start": (
                str(candidate_index[0].date()) if len(candidate_index) else None
            ),
            "last_candidate_start": (
                str(candidate_index[-1].date()) if len(candidate_index) else None
            ),
        }
        results = {}
        close = self.close.to_numpy(dtype=float)
        steps = np.arange(1, self.horizon_days + 1)
        for variant_name, group in normalized.items():
            details = common_details.copy()
            if not len(candidates):
                results[variant_name] = self._empty(
                    variant_name, "No completed historical forward windows", **details,
                )
                continue
            mask = np.ones(len(candidates), dtype=bool)
            columns = {}
            criterion_details = []
            warning = None
            for c in group:
                series = self._indicator(c)
                current = float(series.iloc[-1])
                as_of_reference = isinstance(c, AsOfCriterion)
                reference = current if as_of_reference else float(c.value)
                if not np.isfinite(reference):
                    warning = f"As-of indicator {c.indicator}_{c.window} unavailable"
                    break
                vals = series.iloc[candidates].to_numpy(dtype=float)
                if c.match == "within":
                    if c.tolerance_mode == "relative":
                        if reference == 0:
                            raise ValueError(
                                f"{c.indicator}_{c.window}: relative tolerance "
                                "cannot be used with zero reference; "
                                "use tolerance_mode='absolute'."
                            )
                        margin = float(c.tolerance) * abs(reference)
                    else:
                        margin = float(c.tolerance)
                    lower, upper = reference - margin, reference + margin
                    matches = (vals >= lower) & (vals <= upper)
                elif c.match == "below":
                    lower, upper = None, reference
                    matches = vals <= reference
                else:
                    lower, upper = reference, None
                    matches = vals >= reference
                mask &= np.isfinite(vals) & matches
                name = f"{c.indicator}_{c.window}"
                key, suffix = name, 2
                while key in columns:
                    key = f"{name}_{suffix}"
                    suffix += 1
                columns[key] = vals
                source = "as_of_date" if as_of_reference else "fixed_value"
                tolerance_mode = c.tolerance_mode if c.match == "within" else None
                details.update({
                    f"current_{key}": current,
                    f"reference_{key}": reference,
                    f"reference_source_{key}": source,
                    f"match_{key}": c.match,
                    f"tolerance_{key}": c.tolerance,
                    f"tolerance_mode_{key}": tolerance_mode,
                    f"lower_bound_{key}": lower,
                    f"upper_bound_{key}": upper,
                })
                criterion_details.append({
                    "indicator": c.indicator,
                    "window": c.window,
                    "reference_source": source,
                    "current": current,
                    "reference": reference,
                    "match": c.match,
                    "tolerance": c.tolerance,
                    "tolerance_mode": tolerance_mode,
                    "lower_bound": lower,
                    "upper_bound": upper,
                })
            details["criteria"] = criterion_details
            if warning is not None:
                results[variant_name] = self._empty(variant_name, warning, **details)
                continue

            selected = candidates[mask]
            matched_index = self.close.index[selected]
            n_matched = len(selected)
            nonoverlap = 0
            last_selected = -self.horizon_days
            for position in selected:
                if nonoverlap == 0 or position >= last_selected + self.horizon_days:
                    nonoverlap += 1
                    last_selected = int(position)
            year_counts = matched_index.year.value_counts() if n_matched else None
            details.update({
                "matched_dates": n_matched,
                "nonoverlapping_paths": nonoverlap,
                "max_year_share": (
                    float(year_counts.max() / n_matched) if n_matched else np.nan
                ),
                "first_matched_start": (
                    str(matched_index[0].date()) if n_matched else None
                ),
                "last_matched_start": (
                    str(matched_index[-1].date()) if n_matched else None
                ),
            })
            if not n_matched:
                results[variant_name] = PathSample(
                    variant_name, self.reference_price, self.horizon_days,
                    np.empty((0, self.horizon_days)), details=details,
                    status="no_matches", start_dates=matched_index,
                )
                continue
            paths = (
                self.reference_price * close[selected[:, None] + steps]
                / close[selected, None]
            )
            features = pd.DataFrame({
                key: vals[mask] for key, vals in columns.items()
            })
            results[variant_name] = PathSample(
                variant_name, self.reference_price, self.horizon_days,
                paths, details=details,
                start_dates=matched_index, features=features,
            )
        return self._results("Conditional empirical", results)

    def backtest(
        self, *, method: str = "empirical", threshold: float = -10.0,
        pct: bool = True, mode: str = "mixed", price_type: str = "terminal",
        test_start: str = "2018-01-01", test_end: Optional[str] = None,
        step_days: int = 63, method_kwargs: Optional[dict] = None,
        variant: Optional[str] = None,
    ) -> BacktestResult:
        """Walk-forward calibration of the same event as threshold_probabilities().

        The default is terminal return < -10%; use min/below or max/above
        for future DAILY-CLOSE barrier events (not intraday touch).
        Forecast inputs are truncated at each historical anchor; realized
        outcomes use the next horizon_days actual closing prices. Overlapping
        test windows are dependent; Brier scores alone do not prove calibration.
        threshold is a percentage-point change if pct=True (e.g. -10 = -10%).
        """
        if method not in BACKTEST_METHODS:
            raise ValueError("Unknown method")
        _validated_thresholds([threshold], pct=pct, mode=mode)
        if price_type not in ("terminal", "min", "max"):
            raise ValueError("price_type must be 'terminal', 'min', or 'max'")
        if (not isinstance(step_days, (int, np.integer))
                or isinstance(step_days, (bool, np.bool_)) or step_days < 1):
            raise ValueError("step_days must be a positive integer")
        kwargs = dict(method_kwargs or {})
        idx = self.close.index
        max_start_index = len(idx) - 1 - self.horizon_days
        first_index = int(idx.searchsorted(pd.Timestamp(test_start)))
        if test_end is not None:
            max_start_index = min(
                max_start_index,
                int(idx.searchsorted(pd.Timestamp(test_end), side="right")) - 1,
            )
        rows = []
        for i in range(first_index, max_start_index + 1, int(step_days)):
            date = idx[i]
            base = float(self.close.iloc[i])
            level = _threshold_level(float(threshold), reference=base, pct=pct)
            is_above = _threshold_is_above(
                float(threshold), reference=base, pct=pct, mode=mode,
            )
            future = self.close.iloc[i + 1:i + 1 + self.horizon_days].to_numpy(dtype=float)
            realized_value = float(
                future[-1] if price_type == "terminal"
                else future.min() if price_type == "min" else future.max()
            )
            realized_event = int(_threshold_event(realized_value, level=level, above=is_above))
            pred, status, sample_count, used_variant = np.nan, "insufficient_data", 0, variant
            try:
                engine = type(self)(
                    self.prices, as_of_date=str(date.date()),
                    horizon_days=self.horizon_days,
                )
                results = getattr(engine, method)(**kwargs)
                if used_variant is None:
                    if len(results.variants) != 1:
                        raise ValueError("Specify variant for methods with multiple configurations")
                    used_variant = next(iter(results.variants))
                if used_variant not in results.variants:
                    raise ValueError(f"Unknown variant: {used_variant}")
                chosen = results.variants[used_variant]
                sample_count = chosen.n_paths
                status = chosen.status
                if status == "ok" and sample_count:
                    table = results.threshold_probabilities(
                        thresholds=[threshold], pct=pct, mode=mode,
                        price_type=price_type, variant=used_variant, styled=False,
                    )
                    pred = float(table.iloc[0, 0]) / 100.0
            except (ValueError, KeyError) as exc:
                status = f"error: {exc}"
            rows.append({
                "date": date, "price": base, "threshold_price": level,
                "realized_value": realized_value,
                "predicted_probability": pred, "realized_event": realized_event,
                "status": status, "variant": used_variant,
                "n_paths": sample_count,
            })
        return BacktestResult(
            method, float(threshold), bool(pct), mode, price_type,
            self.horizon_days, pd.DataFrame(rows, columns=[
                "date", "price", "threshold_price", "realized_value",
                "predicted_probability", "realized_event", "status", "variant",
                "n_paths",
            ]),
        )



def _validate_simulations(n: int) -> None:
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("simulations must be a positive integer")





class ModelComparison:
    """Compare four price-path methods under common event definitions.

    Results are model-conditional, not measures of predictive accuracy.
    Normal is a zero-log-drift volatility-risk baseline; historical methods
    differ in drift, sampling, dependence, and effective sample size.
    Users are responsible for interpreting these differences.

    Omitted method configurations use defaults; None excludes a method.
    When every method is None, construction returns None.

    Comparison tables reuse stored ModelResults without regenerating paths.
    """

    def __new__(
        cls, *args,
        normal=_DEFAULT_MODEL,
        empirical=_DEFAULT_MODEL,
        bootstrap=_DEFAULT_MODEL,
        conditional_empirical=_DEFAULT_MODEL,
        **kwargs,
    ):
        if all(
            config is None
            for config in (normal, empirical, bootstrap, conditional_empirical)
        ):
            return None
        return super().__new__(cls)

    def __init__(
        self, prices: pd.DataFrame, *,
        horizon_days: int,
        as_of_date: Optional[str] = None,
        history_start: Optional[str] = None,
        min_history_days: Optional[int] = None,
        simulations: Optional[int] = None,
        random_seed: Optional[int] = None,
        normal=_DEFAULT_MODEL,
        empirical=_DEFAULT_MODEL,
        bootstrap=_DEFAULT_MODEL,
        conditional_empirical=_DEFAULT_MODEL,
    ):
        self.analysis = PriceTargetAnalysis(
            prices,
            horizon_days=horizon_days,
            as_of_date=as_of_date,
        )

        configs = {
            "normal": normal,
            "empirical": empirical,
            "bootstrap": bootstrap,
            "conditional_empirical": conditional_empirical,
        }

        # Shared arguments apply only to methods that accept them.
        shared_values = {
            "history_start": history_start,
            "min_history_days": min_history_days,
            "simulations": simulations,
            "random_seed": random_seed,
        }
        accepted_shared = {
            "normal": ("simulations", "random_seed"),
            "empirical": (
                "history_start", "min_history_days",
            ),
            "bootstrap": (
                "history_start", "simulations", "random_seed",
            ),
            "conditional_empirical": (
                "history_start",
            ),
        }

        self.results: dict[str, ModelResults] = {}

        for method, config in configs.items():
            if config is None:
                continue

            if config is _DEFAULT_MODEL:
                options = {}
            elif isinstance(config, dict):
                options = config.copy()
            else:
                raise TypeError(
                    f"{method} must be a dict, None, or omitted"
                )

            # Explicit method settings override shared settings.
            for key in accepted_shared[method]:
                value = shared_values[key]
                if value is not None:
                    options.setdefault(key, value)

            self.results[method] = getattr(
                self.analysis, method
            )(**options)

    def threshold_probabilities(
        self, *,
        thresholds: Sequence[float],
        pct: bool = True,
        mode: str = "mixed",
        price_type: str | Sequence[str] = "terminal",
        horizon_days: Optional[int] = None,
        plot: bool = False,
        styled = True
    ) -> Optional[pd.DataFrame]:
        """Combine threshold probabilities with numeric threshold columns.

        All methods use identical threshold, mode, pct, price_type, and
        horizon definitions. The output is indexed by (method, variant).

        plot=False returns a DataFrame.
        plot=True Not supported.
        The threshold comparison visualization is not implemented yet.
        """
        tables = {
            method: result.threshold_probabilities(
                thresholds=thresholds,
                pct=pct,
                mode=mode,
                price_type=price_type,
                horizon_days=horizon_days,
                styled=False,
                summary=False,
            )
            for method, result in self.results.items()
        }

        frame = pd.concat(
            tables,
            names=["method", "variant"] if isinstance(price_type, str)
            else ["method", "variant", "price_type"],
        )

        if plot:
            print(
                "Plotting is not supported for threshold_probabilities(). "
                "Use the returned DataFrame to compare results."
            )

        if styled:
            return (frame.style.format("{:.1f}%", na_rep="—")
            .background_gradient(cmap="Blues", vmin=0, vmax=100, axis=None))
        else:
            return frame

    def return_quantiles(
        self, *, probabilities: Sequence[float], pct: bool = True,
        price_type: str | Sequence[str] = "terminal", horizon_days: Optional[int] = None,
        styled: bool = True,
    ) -> pd.DataFrame | Styler:
        """Compare the same return/price quantiles across stored methods."""
        tables = {
            method: result.return_quantiles(
                probabilities=probabilities, pct=pct, price_type=price_type,
                horizon_days=horizon_days, styled=False, summary=False,
            )
            for method, result in self.results.items()
        }
        frame = pd.concat(
            tables, names=["method", "variant"] if isinstance(price_type, str)
            else ["method", "variant", "price_type"]
        )
        return _quantiles_style(frame, pct=pct) if styled else frame

    def recovery_after_decline(
        self, *, decline_pct: float = 5.0, recovery_pct: float = 0.0,
        horizon_days: Optional[int] = None, styled: bool = True,
    ) -> pd.DataFrame | Styler:
        """Compare post-decline recovery frequencies and timing across methods."""
        tables = {
            method: result.recovery_after_decline(
                decline_pct=decline_pct, recovery_pct=recovery_pct,
                horizon_days=horizon_days, styled=False, summary=False,
            )
            for method, result in self.results.items()
        }
        frame = pd.concat(tables, names=["method", "variant"])
        return _recovery_style(frame) if styled else frame

    def extreme_range(
        self, *,
        probability: float = 0.90,
        horizons: Optional[Sequence[int]] = None,
        plot: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Combine whole-path price envelopes across methods.

        Bounds contain at least the requested fraction of modeled complete
        daily-close paths. They are not terminal-price confidence intervals.

        plot=False returns a DataFrame.
        plot=True displays horizontal lower/upper ranges and returns None.
        """
        frames = []

        for method, result in self.results.items():
            frame = result.extreme_range(
                probability=probability,
                horizons=horizons,
                summary=False,
            ).copy()
            frame.insert(0, "method", method)
            frames.append(frame)

        combined = pd.concat(frames, ignore_index=True)

        if plot:
            self._plot_range_comparison(combined)
            return None

        return combined



    def _plot_range_comparison(self, frame: pd.DataFrame) -> None:
        """Compare whole-path price ranges using horizontal interval bars."""
        valid = frame.loc[
            frame["status"].eq("ok")
            & frame["lower_price"].notna()
            & frame["upper_price"].notna()
        ]
    
        if valid.empty:
            print("ModelComparison: No valid extreme ranges to plot.")
            return
    
        for horizon, group in valid.groupby("horizon_days", sort=True):
            labels = (
                group["method"].astype(str)
                + " / "
                + group["variant"].astype(str)
            ).tolist()
    
            lower = group["lower_price"].to_numpy(dtype=float)
            upper = group["upper_price"].to_numpy(dtype=float)
            y = np.arange(len(group))
    
            fig, ax = plt.subplots(
                figsize=(10, max(3.5, 0.4 * len(group) + 1.5))
            )
    
            reference = self.analysis.reference_price
            
            # Below reference: red
            below_width = np.maximum(0, np.minimum(upper, reference) - lower)
            ax.barh(
                y, width=below_width, left=lower,
                height=0.65, color="indianred", alpha=0.75,
                label="Below reference",
            )
            
            # Above reference: blue
            above_left = np.maximum(lower, reference)
            above_width = np.maximum(0, upper - above_left)
            ax.barh(
                y, width=above_width, left=above_left,
                height=0.65, color="steelblue", alpha=0.75,
                label="Above reference",
            )
    
            ax.axvline(
                self.analysis.reference_price,
                color="dimgray",
                linestyle=":",
                linewidth=1.5,
                label="Reference close",
            )
    
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
    
            # Focus the x-axis on the compared price ranges.
            x_min = min(lower.min(), reference)
            x_max = max(upper.max(), reference)
            padding = max(x_max - x_min, 1.0) * 0.05
            ax.set_xlim(x_min - padding, x_max + padding)
    
            probability = float(group["probability"].iloc[0])
            ax.set(
                title=(
                    f"Model comparison: {probability:.0%} "
                    f"whole-path range ({horizon} trading days)"
                ),
                xlabel="Price / index level",
            )
    
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

