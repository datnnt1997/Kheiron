class EarlyStopping:
    def __init__(self,
                 patience: int = 3,
                 min_delta: float = 0.0):
        self.patience = patience
        self.wait_count = 0
        self.min_delta = min_delta
        self.best_score = None

    def __call__(self, curr_score, *args, **kwargs):
        if abs(self.best_score - curr_score) > self.min_delta:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                return True
        return False

    def restart(self, best_score):
        self.wait_count = 0
        self.best_score = best_score
