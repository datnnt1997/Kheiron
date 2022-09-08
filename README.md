<h2 align="center">Kheiron</h2>

Kheiron: Train and evaluate libary for Pytorch model.

Author: **Trong-Dat Ngo**.

---

## Examples
<details>
<summary>Text classification</summary>
    
```python
options = TrainingOptions(task='text-classification',
                          train_batch_size=8,
                          eval_batch_size=8,
                          metric_for_best_model='macro_f1',
                          greater_is_better=True,
                          track_metrics=True)

trainer = Trainer(model=model,
                  opts=options,
                  train_set=tokenized_sets['train'],
                  eval_set=tokenized_sets['test'],
                  collate_fn=data_collator)

trainer.train()
```

</details>

<details>
<summary>Token classification</summary>

</details>

<details>
<summary>Sammarization</summary>

</details>

<details>
<summary>Translation</summary>

</details>