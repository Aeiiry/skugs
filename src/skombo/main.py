import atexit
import datetime

from loguru import logger as log
from skombo import FD_COLS

from skombo.fd_ops import CharacterManager, FdBotCsvManager, FrameData

START_TIME = datetime.datetime.now()


@atexit.register
def exit_handler() -> None:
    # get the end datetime
    end_time = datetime.datetime.now()

    # get execution time
    elapsed_time = end_time - START_TIME
    log.info(f"Execution time: {elapsed_time} seconds 🤠")


if __name__ == "__main__":

    csv_manager = FdBotCsvManager()

    frame_data = FrameData(csv_manager.pd_data["frame_data"]).clean_fd()

    csv_manager.save_frame_data_csv(frame_data)