class EarlyStopping:
    def __init__(self,
                 patience: int = 3,
                 min_delta: float = 0.0):
        self.patience = patience
        self.wait_count = 0
        self.min_delta = min_delta
        self.best_score = None

    def count(self, curr_score):
        if abs(self.best_score - curr_score) > self.min_delta:
            self.wait_count += 1
