import copy

import pytorch_lightning
import torch
import torch.nn.functional as F
from torch import nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertLayer


def get_model_fn(a):
    def get_model():
        return Jarvis(a=a, model_name=a.model_name, emb_size=300, tokenizer=get_tokenizer(a))

    return get_model


class ModuleJarvis(pytorch_lightning.LightningModule):
    def __init__(
            self,
            base_model,
            lr,
            n_steps,
            train_ds=None,
            val_ds=None,
    ):
        super().__init__()
        self.model = base_model
        self.lr = lr
        self.n_steps = n_steps
        self.train_ds = train_ds
        self.val_ds = val_ds

        for p in self.model.base_model.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        return self.model(**batch)

    def train_dataloader(self):
        return self.train_ds

    def val_dataloader(self):
        return self.val_ds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.n_steps,
            num_warmup_steps=int(0.1 * self.n_steps)
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def get_model(a, train_ds, val_ds):
    base_model = Jarvis(a=a, model_name=a.model_name, emb_size=300, tokenizer=get_tokenizer(a))
    return ModuleJarvis(base_model=base_model, lr=a.learning_rate, n_steps=a.train_steps, train_ds=train_ds,
                        val_ds=val_ds)


def get_tokenizer(a):
    return AutoTokenizer.from_pretrained(a.model_name)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class Jarvis(nn.Module):

    def __init__(self, a, model_name, emb_size, tokenizer):
        super(Jarvis, self).__init__()

        self.untrained = a.untrained

        self.base_model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(a.dropout_rate)
        self.pooler = MeanPooling()
        self.readout = nn.Linear(self.base_model.config.hidden_size, emb_size)
        self.loss = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.max_len = a.max_len
        self.max_skills = a.max_skills
        self.num_heads = a.num_heads
        self.use_skill_weights = a.use_skill_weights

        skill_attention_config = copy.deepcopy(self.base_model.config)
        skill_attention_config.position_embedding_type = None  # Disable pos embeddings

        self.skill_attention = BertLayer(skill_attention_config)
        self.skill_pooling = torch.nn.Parameter(torch.Tensor(self.base_model.config.hidden_size))
        torch.nn.init.uniform_(
            self.skill_pooling,
            a=-1 * self.base_model.config.initializer_range / 2,
            b=self.base_model.config.initializer_range / 2
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-12)

    def tokenize_document(self, document):

        document["skills"] = self.tokenizer(
            document["skills"],
            max_length=self.max_len,
            return_tensors='pt',
            padding="max_length"
        ).data

        return document

    def get_document_embedding(self, documents):

        # Tokenize the documents
        documents = [self.tokenize_document(d) for d in documents]

        # Return untrained model output
        if self.untrained:
            document_skills = [self.base_model(**doc["skills"]).pooler_output for doc in
                               documents]
            return document_skills[0].mean(axis=0).reshape(1, document_skills[0].shape[-1])

        # Get skill embeddings
        document_skills = [self.base_model(**doc["skills"]).last_hidden_state[:, 0] for doc in documents]

        # Pad to a tensor and get attention masks
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

        # Add skill pooling token
        pooling = self.skill_pooling.view(1, 1, -1).repeat(len(documents), 1, 1)
        document_tensor = torch.concat([pooling, document_tensor], dim=1)
        attention_mask = torch.concat([torch.ones(len(documents), 1), attention_mask], dim=1)
        attention_mask = attention_mask.view(len(documents), 1, 1, -1)

        # Get skill interactions via BERT layer (output type is Tuple; take first index to get the Tensor)
        skill_interactions = self.skill_attention(document_tensor, attention_mask=attention_mask)[0]

        # Pool skill interactions (CLS token)
        document_embeddings = skill_interactions[:, 0]

        return document_embeddings

    def forward(self, cv, job, label=None):

        cv_emb = self.get_document_embedding(cv)
        job_emb = self.get_document_embedding(job)

        sim = self.cos(cv_emb, job_emb)
        if label is None:
            return sim
        else:
            loss = contrastive_loss(torch.tensor(label), sim)
            return {"loss": loss}


def contrastive_loss(y, sim, margin=0.5):
    """
    Contrastive loss from Hadsell-et-al.'06
    https://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    dist = 1 - sim
    return torch.mean(y * 2 * dist + (1 - y) * 2 * F.relu(margin - dist))
