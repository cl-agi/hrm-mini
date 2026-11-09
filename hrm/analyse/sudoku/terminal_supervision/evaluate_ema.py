"""Final-only EMA evaluation through the common checkpoint evaluator."""
from hrm.evaluation.checkpoint import main

if __name__ == "__main__":
    main(default_weight_state="ema")
