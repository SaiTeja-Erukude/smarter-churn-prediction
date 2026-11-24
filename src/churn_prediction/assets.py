"""Download and verify the pinned public dataset and encoder checkpoint."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import requests

from .constants import DATA_COMMIT, DATA_SHA256
from .io import digest, write_json
from .paths import ENCODER_MANIFEST


def obtain(url, target, expected_hash):
    target = Path(target)
    if target.is_file() and digest(target) == expected_hash:
        return f"{target.name}: verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    try:
        with requests.get(url, timeout=(30, 180), stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
        if digest(temporary) != expected_hash:
            raise ValueError(f"Downloaded asset checksum differs: {target}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return f"{target.name}: downloaded and verified"


def fetch(args):
    manifest = json.loads(ENCODER_MANIFEST.read_text(encoding="utf-8"))
    jobs = [
        (
            f"https://raw.githubusercontent.com/Suhasnadh/smarter-churn-prediction/{DATA_COMMIT}/Data/Telco_customer_churn.xlsx",
            args.data,
            DATA_SHA256,
        )
    ]
    for entry in manifest["files"]:
        jobs.append(
            (
                f"https://huggingface.co/{manifest['model']}/resolve/{manifest['revision']}/{entry['file']}",
                args.model_dir / entry["file"],
                entry["sha256"],
            )
        )
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda job: obtain(*job), jobs):
            print(result, flush=True)
    write_json(args.model_dir / "manifest.json", manifest)
