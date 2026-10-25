"""Shared fixtures: a hand-written six-row variant, and the bundled synthetic demo data."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from amlcore.config import DEMO_ROOT, Config, load_config
from amlcore.data import build_transactions

# Rows are deliberately out of time order. Bank "021174" checks leading zeros survive.
TINY_TRANS_CSV = """\
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/01 00:10,012,80011F990,021174,800737690,100.00,Euro,100.00,Euro,ACH,1
2022/09/01 00:15,011,800000001,012,800000002,50.00,Yen,50.00,Yen,Cash,1
2022/09/01 00:15,013,800000003,014,800000004,999.00,US Dollar,999.00,US Dollar,Wire,0
2022/09/01 00:06,001,800000005,002,800000006,10.00,UK Pound,12.00,US Dollar,Wire,0
"""

# The SCATTER-GATHER row matches the non-laundering 00:15 transaction above, which must
# still get typology "none". The Yen laundering row is in no block: "unattributed".
TINY_PATTERNS = """\
BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 4 hops
2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
2022/09/01 00:10,012,80011F990,021174,800737690,100.00,Euro,100.00,Euro,ACH,1
END LAUNDERING ATTEMPT - CYCLE

BEGIN LAUNDERING ATTEMPT - SCATTER-GATHER
2022/09/01 00:15,013,800000003,014,800000004,999.00,US Dollar,999.00,US Dollar,Wire,1
END LAUNDERING ATTEMPT - SCATTER-GATHER
"""


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()


@pytest.fixture
def tiny_raw() -> tuple[io.StringIO, str]:
    """The tiny variant's Trans CSV (as a file object) and Patterns text."""
    return io.StringIO(TINY_TRANS_CSV), TINY_PATTERNS


@pytest.fixture
def tiny_txns(config: Config, tiny_raw: tuple[io.StringIO, str]) -> pd.DataFrame:
    return build_transactions(*tiny_raw, config.fx_to_usd)


@pytest.fixture(scope="session")
def demo_txns(config: Config) -> pd.DataFrame:
    """The ~3,100-row synthetic dataset, ingested like the real data (no tail cut)."""
    raw = Path(DEMO_ROOT) / "raw"
    return build_transactions(raw / "DEMO_Trans.csv", (raw / "DEMO_Patterns.txt").read_text(),
                              config.fx_to_usd)
