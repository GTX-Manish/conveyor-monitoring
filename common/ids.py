"""Asset ids and signal names, defined once for the model, the backend, the replayer and scoring.

Simulator zone numbers 1..120 become asset ids 'Z001'..'Z120'; the drive is 'DRIVE'. No other module converts.
"""
N_ZONES = 120
DRIVE = "DRIVE"
ZONE_SIGNALS = ["vibration_rms_mm_s", "vibration_kurtosis", "bearing_temp_c", "acoustic_db", "belt_offset_mm"]
BELT_SIGNALS = ["throughput_tph", "motor_current_a", "motor_temp_c", "gearbox_temp_c",
                "drive_speed_ms", "belt_speed_ms", "slip_pct", "tension_kn"]
RUN_STATES = ["stopped", "ramp", "running"]   # stored as 0, 1, 2
RUNNING = 2


def zone_asset(zone: int) -> str:
    zone = int(zone)
    if not 1 <= zone <= N_ZONES:
        raise ValueError(f"zone {zone} is outside 1..{N_ZONES}")
    return f"Z{zone:03d}"


def asset_zone(asset: str):
    """Zone number for a zone asset, None for the drive."""
    if asset == DRIVE:
        return None
    if len(asset) == 4 and asset[0] == "Z" and asset[1:].isdigit() and 1 <= int(asset[1:]) <= N_ZONES:
        return int(asset[1:])
    raise ValueError(f"unknown asset id {asset!r}")


def simulator_asset(label_asset: str) -> str:
    """Asset id for the simulator's label names ('1'..'120' or 'DRIVE')."""
    return DRIVE if label_asset == DRIVE else zone_asset(int(label_asset))


ASSETS = [zone_asset(z) for z in range(1, N_ZONES + 1)] + [DRIVE]
