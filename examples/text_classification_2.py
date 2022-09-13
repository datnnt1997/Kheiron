from torch import nn

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

from kheiron import Trainer, TrainingOptions
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() and False else "cpu")

tokenizer = get_tokenizer('basic_english')
train_iter, test_iter = AG_NEWS()


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return {'labels': label_list,
            'text': text_list,
            'offsets': offsets
            }


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, labels, offsets):
        embedded = self.embedding(text, offsets)
        criterion = torch.nn.CrossEntropyLoss()
        predicted_label = self.fc(embedded)
        loss = criterion(predicted_label, labels)
        return {'logits': predicted_label, 'loss': loss}


num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class)

train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

options = TrainingOptions(task='text-classification',
                          epochs=10,
                          learning_rate=5.0,
                          train_batch_size=64,
                          eval_batch_size=64,
                          optimizer_name='sgd',
                          no_decay_param_names=[],
                          scheduler_name='steplr',
                          decay_step=1.0,
                          metric_for_best_model='accuracy',
                          greater_is_better=True,
                          early_stopping_steps=3,
                          no_cuda=True)

trainer = Trainer(model=model,
                  opts=options,
                  train_set=train_dataset,
                  eval_set=test_dataset,
                  collate_fn=collate_batch)

trainer.train()
