import torch
from torch import nn as nn

from jarvis2.modeling.encoder import Encoder
from jarvis2.modeling.losses import contrastive_loss, cos_loss


class Comparator(nn.Module):
    """Actual model code containing Jarvis information flow"""

    def __init__(self, a, encoder):
        super(Comparator, self).__init__()

        # validate args
        if a.n_ffn_blocks_emb < 1 or a.n_ffn_blocks_readout < 1:
            raise ValueError(f"Require at least 1 FFN blocks for embeddings and readout")

        if encoder is not None:
            self.cv_encoder = encoder
            self.job_encoder = encoder
        else:
            self.cv_encoder = Encoder(a)
            if a.siamese:
                self.job_encoder = self.cv_encoder
            else:
                # TODO make language model shared between encoders, if frozen
                self.job_encoder = Encoder(a)

            if a.pretrained_path:
                self.cv_encoder.load_state_dict(torch.load(a.pretrained_path))
                if not a.siamese:
                    self.job_encoder.load_state_dict(torch.load(a.pretrained_path))

        if a.loss_fn == "contrastive":
            loss_fn = contrastive_loss
        elif a.loss_fn == "cosine":
            loss_fn = cos_loss
        else:
            raise NotImplementedError(f"Loss function {a.loss_fn} is not implemented")
        self.loss_fn = loss_fn

        if a.pos_label_bias or a.neg_label_bias:
            def loss_fn_wrapper(y, sim):
                y_ = y - a.pos_label_bias * (y == 1)
                y_ = y_ + a.neg_label_bias * (y == 0)
                return loss_fn(y_, sim)

            self.loss_fn = loss_fn_wrapper

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def forward(self, cv, job, label=None):

        cv_emb = self.cv_encoder(cv)
        job_emb = self.job_encoder(job)

        sim = self.cos(cv_emb, job_emb)
        if label is None:
            return sim
        else:
            loss = self.loss_fn(label, sim)
            return {"loss": loss, "sim": sim}
