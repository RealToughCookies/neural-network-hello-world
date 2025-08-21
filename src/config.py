from dataclasses import dataclass, asdict
from typing import Optional, Literal
import json
import pathlib


Device = Literal["cpu", "auto"]


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 128
    subset_train: Optional[int] = 2000
    seed: int = 0
    device: Device = "cpu"
    lr: float = 0.1
    outdir: str = "artifacts"

    def to_json(self, path: str):
        """Save config to JSON file."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: str) -> "TrainConfig":
        """Load config from JSON file."""
        d = json.loads(pathlib.Path(path).read_text())
        return TrainConfig(**d)