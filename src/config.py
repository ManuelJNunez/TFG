"""This module is for sharing variables between training scripts"""
import subprocess
from pathlib import Path

command = ["git", "rev-parse", "--show-toplevel"]

repo_root = subprocess.run(command, capture_output=True, check=True).stdout
repo_root = Path(repo_root[:-1].decode("utf-8"))

artifacts_path = repo_root.joinpath("artifacts")

artifacts_path.mkdir(exist_ok=True)

model_path = artifacts_path.joinpath("model.pth")
