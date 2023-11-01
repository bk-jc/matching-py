from transformers import AutoTokenizer

from modeling.jarvis import Jarvis
from modeling.lightning_model import JarvisLightningWrapper


def get_model(a, train_ds, val_ds):
    base_model = Jarvis(a=a, model_name=a.model_name, tokenizer=get_tokenizer(a))
    return JarvisLightningWrapper(base_model=base_model, lr=a.learning_rate, weight_decay=a.weight_decay,
                                  n_steps=a.train_steps,
                                  n_thresholds=a.n_thresholds, train_ds=train_ds, val_ds=val_ds)


def get_tokenizer(a):
    return AutoTokenizer.from_pretrained(a.model_name)
