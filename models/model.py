import os
from typing import List, Dict
from pathlib import Path
import numpy as np
import math
import torch
import torch.nn as nn
import models.diffusion_process as dp
from models.layers import TransformerCrossAttn, LabelAttention, ContrastiveLearningLoss, MultiViewContrastiveLoss, TransformerEncoder
from pyhealth.medcode import ATC
from pyhealth.metrics import ddi_rate_score
from pyhealth.models import BaseModel
from pyhealth.datasets import SampleEHRDataset
import torch.nn.functional as F

class CD_MDRec(BaseModel):
    def __init__(self,
                 dataset: SampleEHRDataset,
                 embedding_dim,
                 **kwargs
                 ):
        super(CD_MDRec, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.rate = kwargs['ddi_rate']

        self.conditions_embeddings = nn.Embedding(
            self.feat_tokenizers['conditions'].get_vocabulary_size(),
            embedding_dim,
            padding_idx=self.feat_tokenizers['conditions'].get_padding_index(),
        )
        self.procedures_embeddings = nn.Embedding(
            self.feat_tokenizers['procedures'].get_vocabulary_size(),
            embedding_dim,
            padding_idx=self.feat_tokenizers['procedures'].get_padding_index(),
        )
        self.SDNet = SDNet([embedding_dim, embedding_dim], [embedding_dim, embedding_dim], embedding_dim, time_type="cat", norm=True).to(self.device)
        self.DiffProcess = dp.DiffusionProcess('linear-var', 1, 0.0001, 0.01, kwargs['steps'], self.device).to(self.device)
        self.contrast_loss = ContrastiveLearningLoss(0.5)

        self.alpha = kwargs['alpha']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']
        self.samp_step = kwargs['samp_step']
        self.seq_encoder = TransformerCrossAttn(d_model=embedding_dim, nhead=kwargs['heads'],
                                                num_layers=kwargs['num_layers'], dim_feedforward=embedding_dim)
        self.fused_seq_encoder = TransformerEncoder(d_model=embedding_dim, nhead=kwargs['heads'],
                                                   num_layers=kwargs['num_layers'], dim_feedforward=embedding_dim)

        self.label_attention = nn.Linear(embedding_dim * 2, self.label_tokenizer.get_vocabulary_size(),bias=False)

        self.ddi_adj = self.generate_ddi_adj().to(self.device)
        self.weight_cond = nn.Parameter(torch.ones(80)).to(self.device)
        self.weight_proc = nn.Parameter(torch.ones(80)).to(self.device)
        self.map1 = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)
        self.map2 = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)
        BASE_CACHE_PATH = os.path.join(str(Path.home()), ".cache/pyhealth/")
        np.save(os.path.join(BASE_CACHE_PATH, "ddi_adj.npy"), self.ddi_adj)

    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = torch.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        return ddi_adj

    def forward(
            self,
            flag,
            conditions: List[List[List[str]]],
            procedures: List[List[List[str]]],
            drugs: List[List[str]],
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        ''''''
        voc_cond = list({}.fromkeys(sum(sum(conditions, []), [])).keys())
        cond_id = [self.feat_tokenizers["conditions"].vocabulary(voc) for voc in voc_cond]

        voc_proc = list({}.fromkeys(sum(sum(procedures, []), [])).keys())
        proc_id = [self.feat_tokenizers["procedures"].vocabulary(voc) for voc in voc_proc]

        cond_id = torch.tensor(cond_id, dtype=torch.long, device=self.device)   # [sss]
        proc_id = torch.tensor(proc_id, dtype=torch.long, device=self.device)   # [sss]
        cond = self.conditions_embeddings(cond_id)
        proc = self.procedures_embeddings(proc_id)
        if flag == 'train':
            cond_terms = self.DiffProcess.caculate_losses(self.SDNet, cond, True)
            proc_terms = self.DiffProcess.caculate_losses(self.SDNet, proc, True)
            diff_loss = cond_terms["loss"].mean() + proc_terms["loss"].mean()
            self.conditions_embeddings(cond_id).weight = cond_terms["pred_xstart"]   # batch * dim
            self.procedures_embeddings(proc_id).weight = proc_terms["pred_xstart"]
        else:
            samp_cond_xstart = self.DiffProcess.p_sample(self.SDNet, cond, self.samp_step, False)
            samp_proc_xstart = self.DiffProcess.p_sample(self.SDNet, proc, self.samp_step, False)
            self.conditions_embeddings(cond_id).weight = samp_cond_xstart
            self.conditions_embeddings(proc_id).weight = samp_proc_xstart
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)

        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device) # batch * 10 * 39
        conditions = self.conditions_embeddings(conditions) # batch * 10 * 39 * dim
        # condition
        batch_c, visit_c, dis_c, dim_c = conditions.shape
        weight_c = self.weight_cond[:dis_c].view(1,1,-1,1)
        conditions = torch.mul(weight_c, conditions)

        conditions = torch.sum(conditions, dim=2)
        mask = torch.any(conditions != 0, dim=2)

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)  # batch * 10 * 20
        procedures = self.procedures_embeddings(procedures) # batch * 10 * 20 * dim

        # procedures
        batch_p, visit_p, dis_p, dim_p = procedures.shape
        weight_p = self.weight_proc[:dis_p].view(1,1,-1,1)
        procedures = torch.mul(weight_p, procedures)

        procedures = torch.sum(procedures, dim=2)           # batch * 10 * dim

        # 1. way1
        diag_out, proc_out = self.seq_encoder(conditions, procedures, mask) # batch * 10 * dim, batch * 10 * dim
        fused_out1 = self.map1(torch.cat([diag_out, proc_out], dim=2))
        fused_out1 = self.fused_seq_encoder(fused_out1, mask)
        way1 = torch.mean(fused_out1, dim=1)   # batch * dim

        # 2. way2
        side_conditions = self.fused_seq_encoder(conditions, mask)
        side_procedures = self.fused_seq_encoder(procedures, mask)
        side_diag_out, side_proc_out = self.seq_encoder(side_conditions, side_procedures, mask)
        fused_out2 = self.map2(torch.cat([side_diag_out, side_proc_out], dim=2))
        way2 = torch.mean(fused_out2, dim=1) # batch * dim

        way = torch.cat([way1, way2], dim=1) # (way1 + way2) / 2

        mvcl = self.contrast_loss(F.normalize(way1), F.normalize(way2))

        logits = self.label_attention(way)

        curr_drugs = self.prepare_labels(drugs, self.label_tokenizer) # batch * class  multi-hot

        loss = F.binary_cross_entropy_with_logits(logits, curr_drugs)
        y_prob = torch.sigmoid(logits) # batch * class

        y_pred = y_prob.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]    # batch * class
        current_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())

        if current_ddi_rate >= self.rate:
            mul_pred_prob = y_prob.T @ y_prob  # class * class
            batch_ddi_loss = (
                    torch.sum(mul_pred_prob.mul(self.ddi_adj.to(self.device))) / self.ddi_adj.shape[0] ** 2
            )
            loss += self.alpha * batch_ddi_loss
        if flag == 'train':
            loss += diff_loss * self.gamma
        return {
            "loss": loss + self.beta * mvcl,
            "y_prob": y_prob,
            "y_true": curr_drugs,
        }
class SDNet(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(SDNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                        for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
                                         for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

