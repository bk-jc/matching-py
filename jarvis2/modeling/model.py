from jarvis2.modeling.comparator import Comparator
from jarvis2.modeling.lightning import LightningWrapper


def get_model(a, train_ds, val_ds, encoder):
    base_model = Comparator(a, encoder)
    return LightningWrapper(
        base_model=base_model,
        lr=a.learning_rate,
        finetune_lr=a.finetune_lr if a.do_msm else None,
        weight_decay=a.weight_decay,
        n_steps=a.train_steps,
        n_thresholds=a.n_thresholds,
        train_ds=train_ds,
        val_ds=val_ds
    )
