"""Scripts to preprocess data."""

import logging
import subprocess
from collections.abc import Iterable

logger = logging.getLogger(__name__)


def run_c3d(args: Iterable[str]) -> None:
    """Run c3d with the given arguments."""
    cmd_list = ["c3d", *args]
    try:
        output = subprocess.check_output(cmd_list, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        logger.exception("c3d failed: %s", err.output)
    else:
        logger.info(output)



