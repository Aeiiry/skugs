import skug_fd_parse
from skug_fd_parse.frame_data_operations import attempt_to_int


def test_frame_data_operations_main():
    assert skug_fd_parse.frame_data_operations.main() == 0
