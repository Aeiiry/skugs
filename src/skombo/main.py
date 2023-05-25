import skombo
from skombo.fd_ops import FD
from skombo.combo_calc import ComboCalculator, Combo
from loguru import logger as log

combo_calc = ComboCalculator(skombo.TEST_COMBO_CSVS[0])


combo_calc.process_combos()

log.info(FD)
