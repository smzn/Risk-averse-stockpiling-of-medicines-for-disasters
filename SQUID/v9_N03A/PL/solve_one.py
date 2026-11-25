# -*- coding: utf-8 -*-
"""PL モデルで1構成だけ解く。実体は common/driver.py（PL と共通）。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.driver import main   # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:] + ["--granularity", "pl"])
