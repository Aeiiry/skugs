import cProfile
import logging
import os
import pstats

import pytest


@pytest.fixture(scope="session")
def profile():
    from skombo import LOG_DIR

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


def pytest_collection_finish(session):
    """Handle the pytest collection finish hook: configure pyannotate.
    Explicitly delay importing `collect_types` until all tests have
    been collected.  This gives gevent a chance to monkey patch the
    world before importing pyannotate.
    """
    from pyannotate_runtime import collect_types

    collect_types.init_types_collection()


@pytest.fixture(autouse=True)
def collect_types_fixture():
    from pyannotate_runtime import collect_types

    collect_types.start()
    yield
    collect_types.stop()


def pytest_sessionfinish(session, exitstatus):
    from pyannotate_runtime import collect_types

    collect_types.dump_stats("type_info.json")
