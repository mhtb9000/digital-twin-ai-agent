from __future__ import annotations

import json
import re
from typing import Any


def safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Could not parse JSON response.")
