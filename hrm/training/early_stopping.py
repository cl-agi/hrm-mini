"""Early stopping measured in training epochs, not evaluation calls."""
from dataclasses import asdict, dataclass


@dataclass
class EarlyStopping:
    patience_epochs: float
    best_correct: int = -1
    best_epoch: float = 0.0
    best_step: int = 0

    def update(self, correct, epoch, step):
        if correct > self.best_correct:
            self.best_correct = int(correct)
            self.best_epoch = float(epoch)
            self.best_step = int(step)
        return self.should_stop(epoch)

    def should_stop(self, epoch):
        return self.best_correct >= 0 and epoch - self.best_epoch >= self.patience_epochs

    def state_dict(self):
        return asdict(self)
