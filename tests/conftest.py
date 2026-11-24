import json

import pytest

from churn_prediction.paths import DEFAULT_CONFIG


@pytest.fixture
def protocol():
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
