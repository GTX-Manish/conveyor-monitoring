"""Alert rules shared by offline scoring and the live backend, so both count alerts the same way.

An alert opens after `open_after` consecutive windows above the threshold and closes after `close_after`
consecutive windows below it. While the belt is stopped nothing can be learned about a bearing, so the counters
hold: an alert neither opens nor closes during a stop.
"""
import numpy as np

OPEN_AFTER = 2      # (assumption) windows above the threshold before an alert opens: 20 minutes
CLOSE_AFTER = 6     # (assumption) windows below it before the alert closes: 1 hour
ACTIVE_RUN_SHARE = 0.5
# A "predicted failure" call claims the part will fail within this many hours. The classifier behind it is
# trained on 24 h; the claim is deliberately wider (user's decision, the safer side): on the test scenarios only
# half of the calls were followed by a failure within 24 h, and all of them within 36 h.
CALL_HORIZON_HOURS = 72.0
HOLD_P_FAIL = 0.5   # (assumption) an open alert does not close while failure within 24 h is more likely than not
BANDS = [(0.25, "low"), (0.5, "medium"), (0.75, "high"), (float("inf"), "critical")]


class Debouncer:
    """Open/close state for a fixed list of assets, advanced one window at a time."""

    def __init__(self, n_assets: int, open_after: int = OPEN_AFTER, close_after: int = CLOSE_AFTER):
        self.open_after, self.close_after = open_after, close_after
        self.above_run = np.zeros(n_assets, dtype=np.int32)
        self.below_run = np.zeros(n_assets, dtype=np.int32)
        self.is_open = np.zeros(n_assets, dtype=bool)

    def step(self, above, active: bool = True, hold=None):
        """above: (n_assets,) bool. Returns (opened, closed) boolean arrays for this window.

        hold: (n_assets,) bool or None. Where an alert is already open, `hold` keeps it open even if `above` is
        false. It can never open an alert, so it cannot create a false one; it only stops an alert lapsing
        while there is still evidence of failure.
        """
        none = np.zeros_like(self.is_open)
        if not active:
            return none, none
        above = np.asarray(above, dtype=bool)
        if hold is not None:
            above = above | (self.is_open & np.asarray(hold, dtype=bool))
        self.above_run = np.where(above, self.above_run + 1, 0)
        self.below_run = np.where(above, 0, self.below_run + 1)
        opened = ~self.is_open & (self.above_run >= self.open_after)
        closed = self.is_open & (self.below_run >= self.close_after)
        self.is_open = (self.is_open | opened) & ~closed
        return opened, closed


def alert_spans(above, active, open_after: int = OPEN_AFTER, close_after: int = CLOSE_AFTER, hold=None) -> list:
    """Offline replay of the rules. above: (W, A) bool, active: (W,) bool -> [(asset, open_w, close_w)].

    close_w is None for an alert still open at the end. hold: (W, A) bool or None, see `Debouncer.step`.
    """
    deb = Debouncer(above.shape[1], open_after, close_after)
    opened_at = {}
    spans = []
    for w in range(above.shape[0]):   # one step per 10-minute window
        opened, closed = deb.step(above[w], bool(active[w]), None if hold is None else hold[w])
        for a in np.flatnonzero(opened):
            opened_at[int(a)] = w
        for a in np.flatnonzero(closed):
            spans.append((int(a), opened_at.pop(int(a)), w))
    spans += [(a, w, None) for a, w in opened_at.items()]
    return sorted(spans, key=lambda s: (s[1], s[0]))


def severity(p_fail: float, eta_hours=None) -> float:
    """0..1. With a time estimate, less time left means more severe; without one, the probability alone."""
    if eta_hours is None or not np.isfinite(eta_hours):
        return float(np.clip(p_fail, 0.0, 1.0))
    return float(np.clip(0.6 * p_fail + 0.4 * (1.0 - min(max(eta_hours, 0.0), 24.0) / 24.0), 0.0, 1.0))


def severity_label(value: float) -> str:
    return next(name for limit, name in BANDS if value < limit)
