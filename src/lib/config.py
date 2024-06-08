from dataclasses import dataclass
import torch


@dataclass
class Variables:
    PRED_THRESHOLD = 0.6

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    TARGETS = {
        0: "Требует ремонта",
        1: "Косметический",
        2: "Евро",
        3: "Дизайнерский"
    }
    EMPTY_TOKEN = 0
    NOT_IN_VOCAB = 1
    LSTM_CONCAT_PARAMS = {
        "vocab_size": 5584 + 2,
        "embedding_dim": 256,
        "hidden_dim": 256,
        "n_layers": 1,
        "use_bidirectional": True
    }
