from typing import Union, List, Dict
from collections import defaultdict


class DefaultMetrics:
    @staticmethod
    def _calc_micro_metrics(tp, p, t, percent=True):
        """
            Compute overall precision, recall and FB1 (default values are 0.0)
            if percent is True, return 100 * original decimal value
        """
        micro_precision = tp / p if p else 0
        micro_recall = tp / t if t else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
            if micro_precision + micro_recall else 0

        if percent:
            return 100 * micro_precision, 100 * micro_recall, 100 * micro_f1
        else:
            return micro_precision, micro_recall, micro_f1

    def _calc_macro_metrics(self, corrects, pred, gold, l_types, percent=True):
        sum_prec, sum_rec, sum_f1 = 0.0, 0.0, 0.0
        for t in l_types:
            prec, rec, f1 = self._calc_micro_metrics(corrects[t], pred[t], gold[t], percent=False)
            sum_prec += prec
            sum_rec += rec
            sum_f1 += f1
        macro_precision, macro_recall, macro_f1 = (sum_prec / len(l_types),
                                                   sum_rec / len(l_types),
                                                   sum_f1 / len(l_types))
        if percent:
            return 100 * macro_precision, 100 * macro_recall, 100 * macro_f1
        else:
            return macro_precision, macro_recall, macro_f1

    def _calc_text_classification_score(self, true_seqs, pred_seqs, percent=True) -> Dict:
        correct_counts = defaultdict(int)
        true_counts = defaultdict(int)
        pred_counts = defaultdict(int)
        for true_tag, pred_tag in zip(true_seqs, pred_seqs):
            if true_tag == pred_tag:
                correct_counts[true_tag] += 1
            true_counts[true_tag] += 1
            pred_counts[pred_tag] += 1
        sum_correct_counts = sum(correct_counts.values())
        sum_true_counts = sum(true_counts.values())
        sum_pred_counts = sum(pred_counts.values())
        types = sorted(list(set(list(true_counts) + list(pred_counts))))
        micro_prec, micro_rec, micro_f1 = self._calc_micro_metrics(sum_correct_counts, sum_pred_counts, sum_true_counts,
                                                                   percent=True)
        macro_prec, macro_rec, macro_f1 = self._calc_macro_metrics(correct_counts, pred_counts, true_counts,
                                                                   l_types=types,
                                                                   percent=True)
        acc = sum_correct_counts/len(true_seqs)
        if percent:
            acc = acc * 100
        return {
            'accuracy': acc,
            'micro_precision': micro_prec, 'micro_recall': micro_rec, 'micro_f1': micro_f1,
            'macro_precision': macro_prec, 'macro_recall': macro_rec, 'macro_f1': macro_f1,
        }

    def __call__(self,
                 true_seqs: List[Union[str, int]],
                 pred_seqs: List[Union[str, int]],
                 task: str,
                 percent: bool = True, *args, **kwargs) -> Dict:
        if task == 'text-classification':
            return self._calc_text_classification_score(true_seqs, pred_seqs, percent)
        else:
            raise ValueError(
                f"No suitable metrics found for task `{task}` or task name not valid."
            )