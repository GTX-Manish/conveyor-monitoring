import pytest

from simulator.config import config_from_dict
from simulator.generate import generate

NO_MESS = {"missing_rate": 0, "outages_per_30d": 0, "stuck_per_30d": 0}
NO_FAULTS = {"idler_bearing": {"per_30d": 0}, "drive_overheat": {"per_30d": 0}}


@pytest.fixture(scope="session")
def cfg14():
    """Healthy 14-day baseline (no faults), used by the stage 1 checks."""
    return config_from_dict({"scenario": "test14", "seed": 1, "days": 14, "faults": NO_FAULTS, "messiness": NO_MESS})


@pytest.fixture(scope="session")
def scn14(cfg14):
    return generate(cfg14)
