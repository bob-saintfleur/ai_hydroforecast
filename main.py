# Copyright 2025 Gustave Eiffel University
# Licensed under the EUROPEAN UNION PUBLIC LICENCE v. 1.2
# See the LICENSE file in the project root for full license information.

# !/bin/env python3

from utils.logger import logger
from datetime import datetime
from utils.run_model import launch_console
from utils.args_getter import get_run_args


def run():
    """Run for all modes except climatology"""
    start = datetime.now()
    logger.info('=' * 35)
    logger.info(f'STARTED (NEW RUNS)')

    u_cfg = get_run_args()
    logger.info(f"Run mode : {u_cfg.run_mode.upper()}")

    launch_console(u_cfg)
    now = datetime.now()
    logger.info(f'TOTAL DURATION = {now - start}')


if __name__ == '__main__':
    run()
