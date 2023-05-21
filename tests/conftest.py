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


