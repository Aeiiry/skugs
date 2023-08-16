import pytest
from pyannotate_runtime import collect_types

from skombo.combo_calc import ComboCalculator


collect_types.init_types_collection()

from loguru import logger as log

from skombo.utils import expand_all_x_n
import skombo

