from time import sleep
import skombo
from skombo import fd_ops, combo_calc, utils


def main():
    fd = fd_ops.FD
    fd.to_csv("fd_cleaned.csv")
    sleep(1)

if __name__ == "__main__":
    main()