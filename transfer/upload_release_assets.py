"""Resume model-asset uploads to the private CV-1 release using Git credentials."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import requests


REPO = "andrei434566-rgb/CV-1"
TAG = "cv1-workspace-v1"
OUT = Path(__file__).resolve().parent / "out"


def github_token() -> str:
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return values["password"]


def main() -> None:
    manifest = json.loads((OUT / "cv1-runtime-manifest.json").read_text(encoding="utf-8"))
    names = ["cv1-runtime-manifest.json", manifest["runtime_archive"]["name"]]
    names.extend(part["name"] for part in manifest["parts"])
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {github_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "CV-1-model-transfer",
    })
    release_response = session.get(
        f"https://api.github.com/repos/{REPO}/releases?per_page=100", timeout=30
    )
    release_response.raise_for_status()
    release = next((item for item in release_response.json() if item["tag_name"] == TAG), None)
    if release is None:
        raise RuntimeError(f"Release {TAG} not found")
    assets_response = session.get(
        f"https://api.github.com/repos/{REPO}/releases/{release['id']}/assets?per_page=100",
        timeout=30,
    )
    assets_response.raise_for_status()
    existing = {asset["name"]: asset for asset in assets_response.json()}
    upload_url = release["upload_url"].split("{", 1)[0]
    for name in names:
        path = OUT / name
        size = path.stat().st_size
        if name in existing:
            if existing[name].get("state") != "uploaded":
                raise RuntimeError(f"Incomplete release asset needs inspection before retry: {name}")
            if existing[name]["size"] != size:
                raise RuntimeError(f"Existing asset has a different size: {name}")
            print(f"Already uploaded: {name} ({size} bytes)", flush=True)
            continue
        print(f"Uploading: {name} ({size} bytes)", flush=True)
        with path.open("rb") as stream:
            response = session.post(
                upload_url,
                params={"name": name},
                headers={"Content-Type": "application/octet-stream", "Content-Length": str(size)},
                data=stream,
                timeout=(30, 7200),
            )
        if response.status_code != 201:
            raise RuntimeError(f"Upload failed for {name}: HTTP {response.status_code}: {response.text[:500]}")
        result = response.json()
        if result.get("size") != size:
            raise RuntimeError(f"GitHub returned a different size for {name}")
        print(f"Uploaded: {name}", flush=True)


if __name__ == "__main__":
    main()
