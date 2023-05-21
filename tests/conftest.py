import os
import pytest
import logging
from skombo import get_logger
from skombo import LOG_DIR
import cProfile
import pstats


@pytest.fixture(scope="session")
def LOG() -> logging.Logger:
    return get_logger()


@pytest.fixture(scope="session")
def profile():
    profile = cProfile.Profile()
    profile.enable()
    yield profile
    profile.disable()
    profile.create_stats()
    stats = pstats.Stats(profile) if profile else None
    if stats:
        stats.sort_stats("cumulative")
        stats.print_stats(15)
        stats.dump_stats(os.path.join(LOG_DIR, "test.prof"))
