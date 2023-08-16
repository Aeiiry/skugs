import atexit
import datetime

from loguru import logger as log

from skombo.fd_ops import frame_data

START_TIME = datetime.datetime.now()


@atexit.register
def exit_handler() -> None:
    # get the end datetime
    end_time = datetime.datetime.now()

    # get execution time
    elapsed_time = end_time - START_TIME
    log.info(f"Execution time: {elapsed_time} seconds 🤠")


log.info("Done!")
