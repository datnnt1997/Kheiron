import json


class TrainingStats:
    def __init__(self):
        default_stats = {
            # Training progress
            'curr_epoch': None,
            'curr_global_step': None,
            'max_steps': 0,
            'train_examples': 0,
            'train_epochs': 0,
            'train_batch_size': 0,
            'train_loss': None,
            # Evaluate progress
            'eval_examples': 0,
            'evaluation_strategy': None,
            'best_score': None,
            'best_loss': None,
            'best_step': None,

        }
        self._stats = default_stats

    def set_stats(self, stats):
        self._stats = stats

    def get_value(self, key, default=None):
        return self._stats.get(key, default)

    def set_value(self, key, value):
        self._stats[key] = value

    def inc_value(self, key, count=1, start=0):
        d = self._stats
        if d.setdefault(key, start) is None:
            d[key] = count
        else:
            d[key] = d.setdefault(key, start) + count

    def clear_stats(self):
        self._stats.clear()

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(self._stats, indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    def load_from_json(self, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        self._stats = json.loads(text)