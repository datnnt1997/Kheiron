from datasets import load_dataset
from kheiron import Trainer, TrainingOptions

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification


def preprocess_func(examples):
    tokenized_example = tokenizer(examples['text'], truncation=True)
    tokenized_example['labels'] = [id2label.index(l) for l in examples['labels']]
    return tokenized_example


rawsets = load_dataset('papluca/language-identification')
id2label = sorted(set(rawsets['train']['labels']))
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
tokenized_sets = rawsets.map(preprocess_func, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=len(id2label))

options = TrainingOptions(task='text-classification',
                          train_batch_size=32,
                          eval_batch_size=32,
                          metric_for_best_model='macro_f1',
                          greater_is_better=True)

trainer = Trainer(model=model,
                  args=options,
                  train_set=tokenized_sets['train'],
                  eval_set=tokenized_sets['test'],
                  collate_fn=data_collator)

trainer.train()
