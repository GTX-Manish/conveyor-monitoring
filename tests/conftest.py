import pytest

from simulator.config import Config
from simulator.generate import generate


@pytest.fixture(scope="session")
def cfg14():
    return Config(scenario="test14", seed=1, days=14)


@pytest.fixture(scope="session")
def scn14(cfg14):
    return generate(cfg14)
