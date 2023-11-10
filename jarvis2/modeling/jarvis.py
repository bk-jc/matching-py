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
        self.dropout = nn.Dropout(a.dropout_rate)
        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.max_len = a.max_len
        self.max_skills = a.max_skills
        self.num_heads = a.num_heads
        self.cache_embeddings = a.cache_embeddings
        self.skill_prefix = a.skill_prefix

        skill_attention_config = copy.deepcopy(self.base_model.config)
        setattr(skill_attention_config, "hidden_size", a.hidden_dim)
        skill_attention_config.position_embedding_type = None  # Disable pos embeddings

        if a.pooling_mode == "cls":
            # TODO configurable config.num_attention_heads config.intermediate_size config.
            self.skill_attention = BertLayer(skill_attention_config).attention
            self.skill_pooling = torch.nn.Parameter(torch.Tensor(a.hidden_dim))
            torch.nn.init.uniform_(
                self.skill_pooling,
                a=-1 * self.base_model.config.initializer_range / 2,
                b=self.base_model.config.initializer_range / 2
            )

        if a.loss_fn == "contrastive":
            self.loss_fn = contrastive_loss
        elif a.loss_fn == "cosine":
            self.loss_fn = cos_loss

        self.ffn_emb = FFN(
            input_dim=self.base_model.config.hidden_size,
            output_dim=a.hidden_dim,
            hidden_dim=a.hidden_dim,
            n_blocks=a.n_ffn_blocks_emb,
            dropout_rate=a.dropout_rate,
        )
        self.ffn_readout = FFN(
            input_dim=a.hidden_dim if a.n_ffn_blocks_emb else self.base_model.config.hidden_size,
            output_dim=a.readout_dim,
            hidden_dim=a.hidden_dim,
            n_blocks=a.n_ffn_blocks_emb,
            dropout_rate=a.dropout_rate,
        )

        if self.cache_embeddings:
            cache_file = f'{self.base_model.config.name_or_path.replace("/", "-")}.pt'
            self.cache_path = Path(__file__).parent.parent.parent / ".cache"
            os.makedirs(self.cache_path, exist_ok=True)
            if os.path.exists(self.cache_path / cache_file):
                self.cache = torch.load(self.cache_path / cache_file)
            else:
                self.cache = {}

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def get_document_embedding(self, documents):

        document_tensor, attention_mask = [], []
        for doc in documents:
            tensor, attn_mask = self.get_skill_embeddings(doc)
            document_tensor.append(tensor)
            attention_mask.append(attn_mask)
        document_tensor, attention_mask = torch.stack(document_tensor), torch.stack(attention_mask)

        batch_size, n_skills = document_tensor.shape[:2]

        # Dense layer
        document_tensor = document_tensor.view(batch_size * n_skills, -1)
        document_tensor = self.ffn_emb(document_tensor)
        document_tensor = document_tensor.view(batch_size, n_skills, -1)

        if self.pooling_mode == "cls":

            # Add skill pooling token
            pooling = self.skill_pooling.view(1, 1, -1).repeat(len(documents), 1, 1)
            document_tensor = torch.concat([pooling, document_tensor], dim=1)
            attention_mask = torch.concat([torch.ones(len(documents), 1), attention_mask], dim=1)
            attention_mask = attention_mask.view(len(documents), 1, 1, -1)

            # Get skill interactions via BERT layer (output type is Tuple; take first index to get the Tensor)
            skill_interactions = self.skill_attention(document_tensor, attention_mask=attention_mask)[0]

            # Pool skill interactions (CLS token)
            document_tensor = skill_interactions[:, 0]

        elif self.pooling_mode == "max":

            mask = attention_mask.unsqueeze(-1).repeat(1, 1, document_tensor.shape[-1])
            document_tensor[mask == 0] = -torch.inf
            document_tensor = torch.max(document_tensor, axis=1).values

        elif self.pooling_mode == "mean":

            mask = attention_mask.unsqueeze(-1).repeat(1, 1, document_tensor.shape[-1])
            document_tensor = document_tensor * mask

            # Compute sum over the sequence_length axis
            sum_pooled = document_tensor.sum(dim=1)

            # Compute the count of non-zero mask values for each position
            non_zero_counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1e-12)  # Avoid division by zero
            document_tensor = sum_pooled / non_zero_counts

        # Dense layer
        document_tensor = self.ffn_readout(document_tensor)

        return document_tensor

    def get_skill_embeddings(self, document):

        if self.cache_embeddings:
            skills = []
            for skill in document["skills"]:
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
            document["skills"] = self.tokenizer(
                document["skills"],
                max_length=self.max_len,
                return_tensors='pt',
                padding="max_length"
            ).data
            skills = self.base_model(**document["skills"]).last_hidden_state[:, 0]

        # Pad tensor and get attention masks
        pad_length = self.max_skills
        attention_mask = pad(torch.ones(skills.shape[0]), (0, pad_length - skills.shape[0]))
        skills = pad(skills, (0, 0, 0, pad_length - skills.shape[0]))

        return skills, attention_mask

    def forward(self, cv, job, label=None):

        cv_emb = self.get_document_embedding(cv)
        job_emb = self.get_document_embedding(job)

        sim = self.cos(cv_emb, job_emb)
        if label is None:
            return sim
        else:
            loss = self.loss_fn(label, sim)
            return {"loss": loss, "sim": sim}
