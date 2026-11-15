"""Turn the model's strongest features into sentences a person can read.

The model's evidence is a list of (feature name, value) pairs such as ('bearing_temp_c__dev6h', 9.8). This module
says what that means: "Bearing temperature is 9.8 °C higher than the other zones on its run (6 h average)".
Only features that describe the equipment are explained; context features (share of time running, position on
the belt and so on) are skipped, because they are not something the model "saw go wrong".
"""
import math

# signal -> (name, unit, decimals). Vibration RMS is compared as a ratio, so it is reported in percent.
SIGNALS = {
    "bearing_temp_c": ("Bearing temperature", "°C", 1), "acoustic_db": ("Acoustic level", "dB", 1),
    "vibration_rms_mm_s": ("Vibration", "%", 0), "vibration_kurtosis": ("Vibration kurtosis (impact spikes)", "", 2),
    "belt_offset_mm": ("Belt sideways offset", "mm", 1), "motor_temp_c": ("Motor temperature", "°C", 1),
    "gearbox_temp_c": ("Gearbox temperature", "°C", 1), "motor_current_a": ("Motor current", "A", 0),
    "tension_kn": ("Belt tension", "kN", 0), "slip_pct": ("Belt slip", "%", 2),
    "throughput_tph": ("Throughput", "t/h", 0), "belt_speed_ms": ("Belt speed", "m/s", 2),
}
WINDOW = {"dev10": "last 10 minutes", "dev1h": "1 h average", "dev6h": "6 h average", "dev24h": "24 h average"}
DRIVE_TEMPS = {"motor_temp_c", "gearbox_temp_c"}


def _amount(signal: str, value: float) -> str:
    name, unit, decimals = SIGNALS[signal]
    if signal == "vibration_rms_mm_s":          # the feature is a difference of logarithms: turn it into percent
        value = (math.exp(value) - 1.0) * 100.0
    number = f"{abs(value):.{decimals}f}"
    return f"{number} {unit}".strip() if unit != "%" else f"{number}%"


def explain(name: str, value) -> str | None:
    """One sentence for one feature, or None if the feature is context and not a symptom."""
    if value is None or not math.isfinite(value):
        return None
    if name == "motor_minus_gearbox_1h":
        return f"The motor is {abs(value):.1f} °C {'hotter' if value >= 0 else 'cooler'} than the gearbox (1 h average)"
    if name.startswith("current_per_load"):
        return "The motor is drawing an unusual amount of current for the load it carries"
    signal, _, kind = name.partition("__")
    if signal not in SIGNALS or not kind:
        return None
    label, amount = SIGNALS[signal][0], _amount(signal, value)
    up = value >= 0
    if kind in WINDOW:
        if signal in DRIVE_TEMPS:                # the drive is compared with the idler bearings, which follow the weather
            return f"{label} is {amount} {'above' if up else 'below'} the idler bearings' temperature ({WINDOW[kind]})"
        return f"{label} is {amount} {'higher' if up else 'lower'} than the other zones on its run ({WINDOW[kind]})"
    if kind in ("rise1h", "rise6h"):
        span = "last hour" if kind == "rise1h" else "last 6 hours"
        return f"{label} has {'risen' if up else 'fallen'} {amount} in the {span}, compared with its own last 24 hours"
    if kind == "nb6h":
        return f"{label} is {amount} {'higher' if up else 'lower'} than the two neighbouring zones (6 h average)"
    if kind == "peak1h":
        return f"{label} peaked {amount} {'above' if up else 'below'} the other zones' peaks in the last hour"
    if kind == "spread1h":
        return f"{label} is fluctuating {'more' if up else 'less'} than in the other zones (last hour)"
    if kind.startswith("mean"):
        return None                              # a plain level, not a comparison: not a symptom on its own
    return None


def reasons(features: list, limit: int = 3) -> list:
    """Sentences for the strongest explainable features, strongest first."""
    out = []
    for f in features:
        text = explain(f["name"], f.get("value"))
        if text and text not in out:
            out.append(text)
        if len(out) == limit:
            break
    return out
