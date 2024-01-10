import os
from pathlib import Path

import torch
from torch import nn as nn
from torch._C._nn import pad
from transformers import AutoConfig, BertLayer, AutoModel, AutoTokenizer

from modeling.layers import FFN


class Encoder(nn.Module):
    """Encode a document to a vector"""

    def __init__(self, a):
        super(Encoder, self).__init__()
        self.max_len = a.max_len
        self.max_skills = a.max_skills
        self.num_heads = a.num_heads
        self.cache_embeddings = a.cache_embeddings
        self.skill_prefix = a.skill_prefix
        self.use_jobtitle = a.use_jobtitle
        self.alpha = a.alpha
        self.pooling_mode = a.pooling_mode
        self.model_name = a.model_name
        self.readout_dim = a.readout_dim

        self.tokenizer = AutoTokenizer.from_pretrained(a.model_name)
        self.base_model = AutoModel.from_pretrained(a.model_name)
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.ffn_emb = self.get_ffn(a)
        self.ffn_readout = self.get_ffn(a, readout=True)

        # TODO move to pooling module
        if self.pooling_mode == "cls" or self.alpha > 0:
            skill_attention_config = AutoConfig.from_pretrained(a.model_name)
            setattr(skill_attention_config, "hidden_size", a.hidden_dim)
            skill_attention_config.position_embedding_type = None  # Disable pos embeddings
            skill_attention_config.num_attention_heads = a.num_heads
            self.skill_attention_layer = BertLayer(skill_attention_config).attention

            self.skill_cls = torch.nn.Parameter(torch.Tensor(a.hidden_dim)).view(1, 1, -1)
            torch.nn.init.uniform_(
                self.skill_cls,
                a=-1 * skill_attention_config.initializer_range / 2,
                b=skill_attention_config.initializer_range / 2
            )

        if self.cache_embeddings:
            cache_file = f'{self.base_model.config.name_or_path.replace("/", "-")}.pt'
            cache_path = Path(__file__).parent.parent.parent / ".cache"
            os.makedirs(cache_path, exist_ok=True)
            cache_path = cache_path / cache_file
            if os.path.exists(cache_path):
                self.cache = torch.load(cache_path)
            else:
                self.cache = {}

    def forward(self, documents, return_pre_pooled=False):
        document_tensor, skill_mask = self.get_tensors_and_masks(documents)
        batch_size, n_skills = document_tensor.shape[:2]

        # Dense layer with reshaping for skills
        document_tensor = document_tensor.view(batch_size * n_skills, -1)
        document_tensor = self.ffn_emb(document_tensor)
        document_tensor = document_tensor.view(batch_size, n_skills, -1)

        if self.pooling_mode == "cls" or self.alpha > 0:
            attention_tensor = self.skill_attention(skill_mask, document_tensor, documents)
        else:
            attention_tensor = None

        if return_pre_pooled:
            return attention_tensor

        # Pooling out skill dimension
        document_tensor = self.pool_skills(skill_mask, document_tensor, attention_tensor)

        # Dense layer
        document_tensor = self.ffn_readout(document_tensor)

        return document_tensor

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

    def get_ffn(self, a, readout=False):
        return FFN(
            input_dim=a.hidden_dim if (readout and a.n_ffn_blocks_emb) else self.base_model.config.hidden_size,
            output_dim=a.readout_dim if readout else a.hidden_dim,
            hidden_dim=a.hidden_dim,
            n_blocks=a.n_ffn_blocks_emb,
            dropout_rate=a.dropout_rate,
            relu_on_last_layer=readout,
        )

    def get_tensors_and_masks(self, documents):
        document_tensor, attention_mask = [], []
        for doc in documents:
            tensor, attn_mask = self.get_skill_embeddings(doc, self.use_jobtitle)
            document_tensor.append(tensor)
            attention_mask.append(attn_mask)
        return torch.stack(document_tensor), torch.stack(attention_mask)

    def pool_skills(self, skill_mask, document_tensor, attention_tensor):
        if self.pooling_mode == "cls":
            pooled_tensor = self.cls_pooling(attention_tensor)
        elif self.pooling_mode == "max":
            pooled_tensor = self.max_pooling(skill_mask, document_tensor)
        else:  # mean pooling
            pooled_tensor = self.mean_pooling(skill_mask, document_tensor)
        if self.alpha > 0:
            pooled_tensor += self.cls_pooling(attention_tensor)
        return pooled_tensor

    def skill_attention(self, attention_mask, document_tensor, documents):

        # Add skill pooling token
        document_tensor = torch.concat([self.skill_cls.repeat(len(documents), 1, 1), document_tensor], dim=1)
        attention_mask = torch.concat([torch.ones(len(documents), 1), attention_mask], dim=1)
        attention_mask = attention_mask.view(len(documents), 1, 1, -1)

        # Get skill interactions via BERT layer (output type is Tuple; take first index to get the Tensor)
        return self.skill_attention_layer(document_tensor, attention_mask=attention_mask)[0]

    @staticmethod
    def cls_pooling(document_tensor):
        return document_tensor[:, 0]

    @staticmethod
    def max_pooling(attention_mask, document_tensor):
        mask = attention_mask.unsqueeze(-1).repeat(1, 1, document_tensor.shape[-1])
        document_tensor = torch.where(
            mask == 0,
            torch.tensor(-torch.inf).to(document_tensor.device), document_tensor
        )
        return torch.max(document_tensor, dim=1).values

    @staticmethod
    def mean_pooling(attention_mask, document_tensor):
        mask = attention_mask.unsqueeze(-1).repeat(1, 1, document_tensor.shape[-1])
        document_tensor = document_tensor * mask

        # Compute sum over the sequence_length axis
        sum_pooled = document_tensor.sum(dim=1)

        # Compute the count of non-zero mask values for each position
        non_zero_counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1e-12)  # Avoid division by zero
        return sum_pooled / non_zero_counts
