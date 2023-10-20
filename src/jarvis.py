import copy

import torch
from torch import nn as nn
from torch.nn import functional as F
from transformers import AutoModel, BertLayer

from src.losses import contrastive_loss, cos_loss


class Jarvis(nn.Module):
    """Actual model code containing Jarvis information flow"""

    def __init__(self, a, model_name, emb_size, tokenizer):
        super(Jarvis, self).__init__()

        self.untrained = a.untrained
        self.pooling_mode = a.pooling_mode

        self.base_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(a.dropout_rate)
        self.readout = nn.Linear(self.base_model.config.hidden_size, emb_size)  # TODO integrate in model
        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.max_len = a.max_len
        self.max_skills = a.max_skills
        self.num_heads = a.num_heads
        self.use_skill_weights = a.use_skill_weights
        self.cache_embeddings = a.cache_embeddings

        skill_attention_config = copy.deepcopy(self.base_model.config)
        skill_attention_config.position_embedding_type = None  # Disable pos embeddings

        if a.pooling_mode == "cls":
            self.skill_attention = BertLayer(skill_attention_config)
            self.skill_pooling = torch.nn.Parameter(torch.Tensor(self.base_model.config.hidden_size))
            torch.nn.init.uniform_(
                self.skill_pooling,
                a=-1 * self.base_model.config.initializer_range / 2,
                b=self.base_model.config.initializer_range / 2
            )
        elif a.pooling_mode == "max":
            self.dense = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
            self.relu = nn.ReLU()
            self.dense_2 = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)

        self.loss_fn = a.loss_fn
        if a.loss_fn == "contrastive":
            self.loss_fn = contrastive_loss
        elif a.loss_fn == "cosine":
            self.loss_fn = cos_loss

        self.dropout = nn.Dropout(a.dropout_rate)

        if self.cache_embeddings:
            self.cache = {}

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def get_document_embedding(self, documents):

        # Return untrained model output
        if self.untrained:
            raise NotImplementedError(
                "Untrained mode was made broken during parallelization of the forward pass. If this is of interest, "
                "ask Bas for help fixing it. Essentially it requires a function to return pooled document embeddings.")

        document_skills = self.get_doc_embeddings(documents)

        # Pad to a tensor and get attention masks TODO parallelize
        document_tensor = torch.cat([F.pad(doc, (0, 0, 0, self.max_skills - doc.shape[0])).unsqueeze(0)
                                     for doc in document_skills])
        attention_mask = torch.cat([F.pad(torch.ones(doc.shape[0]), (0, self.max_skills - doc.shape[0])).unsqueeze(0)
                                    for doc in document_skills])

        # Multiply embeddings with skill weights
        if self.use_skill_weights:
            skill_weight_tensor = torch.cat(
                [F.pad(torch.tensor([1.] + d["weights"]), (0, self.max_skills - 1 - len(d["weights"]))).unsqueeze(0)
                 for d in documents])
            document_tensor *= skill_weight_tensor.unsqueeze(-1)

        if self.pooling_mode == "cls":

            # Add skill pooling token
            pooling = self.skill_pooling.view(1, 1, -1).repeat(len(documents), 1, 1)
            document_tensor = torch.concat([pooling, document_tensor], dim=1)
            attention_mask = torch.concat([torch.ones(len(documents), 1), attention_mask], dim=1)
            attention_mask = attention_mask.view(len(documents), 1, 1, -1)

            # Get skill interactions via BERT layer (output type is Tuple; take first index to get the Tensor)
            skill_interactions = self.skill_attention(document_tensor, attention_mask=attention_mask)[0]

            # Pool skill interactions (CLS token)
            document_embeddings = skill_interactions[:, 0]

        elif self.pooling_mode == "max":

            # Dense layer  TODO make configurable with custom FFN classes
            shape = document_tensor.shape
            document_tensor = document_tensor.view(-1, *shape[2:])
            document_tensor = self.dense(document_tensor)
            document_tensor = self.relu(document_tensor)
            document_tensor = self.dense_2(document_tensor)
            document_tensor = document_tensor.view(*shape)

            mask = attention_mask.unsqueeze(-1).repeat(1, 1, self.base_model.config.hidden_size)
            document_tensor[mask == 0] = -torch.inf

            document_embeddings = torch.max(document_tensor, axis=1).values

        document_embeddings = self.dropout(document_embeddings)

        return document_embeddings

    def get_doc_embeddings(self, documents):
        document_embeddings = []
        for doc in documents:
            doc_emb = self.get_skill_embeddings(doc)
            document_embeddings.append(doc_emb)
        return document_embeddings

    def get_skill_embeddings(self, document):

        if self.cache_embeddings:
            skills = []
            for skill in document["skills"]:
                if skill in self.cache:
                    skills.append(self.cache[skill])
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

        return skills

    def forward(self, cv, job, label=None):

        cv_emb = self.get_document_embedding(cv)
        job_emb = self.get_document_embedding(job)

        sim = self.cos(cv_emb, job_emb)
        if label is None:
            return sim
        else:
            loss = self.loss_fn(label, sim)
            return {"loss": loss, "sim": sim}
