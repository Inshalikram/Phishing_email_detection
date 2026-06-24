from dataclasses import dataclass

@dataclass
class Config:
    random_state: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    max_features: int = 10000
    max_len: int = 256
    batch_size: int = 16
    epochs: int = 3
    model_name: str = "bert-base-uncased"
    target_names: tuple = ("legitimate", "phishing")

CONFIG = Config()
