import copy
import os
from pathlib import Path

import torch
from torch import nn as nn
from torch.nn.functional import pad
from transformers import AutoModel, BertLayer

from jarvis2.modeling.layers import FFN
from jarvis2.modeling.losses import contrastive_loss, cos_loss


class Jarvis(nn.Module):
    """Actual model code containing Jarvis information flow"""

    def __init__(self, a, model_name, tokenizer):
        super(Jarvis, self).__init__()

        # validate args
        if a.n_ffn_blocks_emb < 1 or a.n_ffn_blocks_readout < 1:
            raise ValueError(f"Require at least 1 FFN blocks for embeddings and readout")

        self.pooling_mode = a.pooling_mode

        self.base_model = AutoModel.from_pretrained(model_name)
        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.max_len = a.max_len
        self.max_skills = a.max_skills
        self.num_heads = a.num_heads
        self.cache_embeddings = a.cache_embeddings
        self.skill_prefix = a.skill_prefix
        self.use_jobtitle = a.use_jobtitle

        if a.pooling_mode == "cls":
            skill_attention_config = copy.deepcopy(self.base_model.config)
            setattr(skill_attention_config, "hidden_size", a.hidden_dim)
            skill_attention_config.position_embedding_type = None  # Disable pos embeddings
            skill_attention_config.num_attention_heads = a.num_heads

            self.skill_attention = BertLayer(skill_attention_config).attention
            self.skill_pooling = torch.nn.Parameter(torch.Tensor(a.hidden_dim))
            torch.nn.init.uniform_(
                self.skill_pooling,
                a=-1 * self.base_model.config.initializer_range / 2,
                b=self.base_model.config.initializer_range / 2
            )

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

        self.ffn_emb_cv = self.get_ffn(a)
        self.ffn_emb_job = self.ffn_emb_cv if a.siamese else self.get_ffn(a)
        self.ffn_readout_cv = self.get_ffn(a, readout=True)
        self.ffn_readout_job = self.ffn_readout_cv if a.siamese else self.get_ffn(a, readout=True)

        if self.cache_embeddings:
            cache_file = f'{self.base_model.config.name_or_path.replace("/", "-")}.pt'
            cache_path = Path(__file__).parent.parent.parent / ".cache"
            os.makedirs(cache_path, exist_ok=True)
            self.cache_path = cache_path / cache_file
            if os.path.exists(self.cache_path):
                self.cache = torch.load(self.cache_path)
            else:
                self.cache = {}

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def get_ffn(self, a, readout=False):
        return FFN(
            input_dim=a.hidden_dim if (readout and a.n_ffn_blocks_emb) else self.base_model.config.hidden_size,
            output_dim=a.readout_dim if readout else a.hidden_dim,
            hidden_dim=a.hidden_dim,
            n_blocks=a.n_ffn_blocks_emb,
            dropout_rate=a.dropout_rate,
            relu_on_last_layer=readout,
        )

    def get_document_embedding(self, documents, ffn_emb, ffn_readout, use_jobtitle=False):

        document_tensor, attention_mask = self.get_tensors_and_masks(documents, use_jobtitle)
        batch_size, n_skills = document_tensor.shape[:2]

        # Dense layer with reshaping for skills
        document_tensor = document_tensor.view(batch_size * n_skills, -1)
        document_tensor = ffn_emb(document_tensor)
        document_tensor = document_tensor.view(batch_size, n_skills, -1)

        # Pooling out skill dimension
        document_tensor = self.pool_skills(attention_mask, document_tensor, documents)

        # Dense layer
        document_tensor = ffn_readout(document_tensor)

        return document_tensor

    def get_tensors_and_masks(self, documents, use_jobtitle):
        document_tensor, attention_mask = [], []
        for doc in documents:
            tensor, attn_mask = self.get_skill_embeddings(doc, use_jobtitle)
            document_tensor.append(tensor)
            attention_mask.append(attn_mask)
        return torch.stack(document_tensor), torch.stack(attention_mask)

    def pool_skills(self, attention_mask, document_tensor, documents):
        if self.pooling_mode == "cls":
            document_tensor = self.cls_pooling(attention_mask, document_tensor, documents)
        elif self.pooling_mode == "max":
            document_tensor = self.max_pooling(attention_mask, document_tensor)
        elif self.pooling_mode == "mean":
            document_tensor = self.mean_pooling(attention_mask, document_tensor)
        return document_tensor

    def cls_pooling(self, attention_mask, document_tensor, documents):

        # Add skill pooling token
        pooling = self.skill_pooling.view(1, 1, -1).repeat(len(documents), 1, 1)
        document_tensor = torch.concat([pooling, document_tensor], dim=1)
        attention_mask = torch.concat([torch.ones(len(documents), 1), attention_mask], dim=1)
        attention_mask = attention_mask.view(len(documents), 1, 1, -1)

        # Get skill interactions via BERT layer (output type is Tuple; take first index to get the Tensor)
        skill_interactions = self.skill_attention(document_tensor, attention_mask=attention_mask)[0]

        # Pool skill interactions (CLS token)
        document_tensor = skill_interactions[:, 0]
        return document_tensor

    @staticmethod
    def max_pooling(attention_mask, document_tensor):
        mask = attention_mask.unsqueeze(-1).repeat(1, 1, document_tensor.shape[-1])
        document_tensor = torch.where(
            mask == 0,
            torch.tensor(-torch.inf).to(document_tensor.device), document_tensor
        )
        return torch.max(document_tensor, axis=1).values

    @staticmethod
    def mean_pooling(attention_mask, document_tensor):
        mask = attention_mask.unsqueeze(-1).repeat(1, 1, document_tensor.shape[-1])
        document_tensor = document_tensor * mask

        # Compute sum over the sequence_length axis
        sum_pooled = document_tensor.sum(dim=1)

        # Compute the count of non-zero mask values for each position
        non_zero_counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1e-12)  # Avoid division by zero
        return sum_pooled / non_zero_counts

    def get_skill_embeddings(self, document, use_jobtitle):

        to_embed = document["skills"]
        if document["jobtitle"] and use_jobtitle:
            to_embed = [document["jobtitle"]] + document["skills"]

        if self.cache_embeddings:
            skills = []
            for skill in to_embed:
                if skill in self.cache:
                    skills.append(self.cache[f"{self.skill_prefix}{skill}"])
                else:
                    tokens = self.tokenizer(
                        skill,
                        max_length=self.max_len,
                        return_tensors='pt',
                        padding="max_length"
                    ).data
                    embedding = self.base_model(**tokens).last_hidden_state[:, 0]
                    self.cache[skill] = embedding
                    skills.append(embedding)

            skills = torch.cat(skills, dim=0)

        else:
            skills = self.tokenizer(
                document["skills"],
                max_length=self.max_len,
                return_tensors='pt',
                padding="max_length"
            ).data
            skills = self.base_model(**skills).last_hidden_state[:, 0]

        # Pad tensor and get attention masks
        pad_length = self.max_skills
        attention_mask = pad(torch.ones(skills.shape[0]), (0, pad_length - skills.shape[0]))
        skills = pad(skills, (0, 0, 0, pad_length - skills.shape[0]))

        return skills, attention_mask

    def forward(self, cv, job, label=None):

        cv_emb = self.get_document_embedding(cv, self.ffn_emb_cv, self.ffn_readout_cv)
        job_emb = self.get_document_embedding(job, self.ffn_emb_job, self.ffn_readout_job,
                                              use_jobtitle=self.use_jobtitle)

        sim = self.cos(cv_emb, job_emb)
        if label is None:
            return sim
        else:
            loss = self.loss_fn(label, sim)
            return {"loss": loss, "sim": sim}
