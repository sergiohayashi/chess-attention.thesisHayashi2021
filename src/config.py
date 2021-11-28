from typing import Dict, Any

config: dict[Any, Any] = {
    'VOCAB_FILE': '../data/vocab/all-labels-shuffle.pgn',
    'TMP_TRAIN_FOLDER': '../temp',
    'CHECKPOINT_FOLDER': '../checkpoints',
    'LOG_FOLDER': '../logs',
    # Setear, para forcar um tamanho diferente
    #    "FORCE_INPUT_SIZE": {
    #        "INPUT_SHAPE": (400, 430),
    #       "ATTENTION_SHAPE": (25, 26),
    #    },
    #
    # "USE_BIG_PLOT": true
}
