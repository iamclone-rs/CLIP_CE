import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self, class_names=None):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.apply(freeze_all_but_bn)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)

        # Optional auxiliary classification loss (CrossEntropy).
        self.cls_loss_weight = float(getattr(self.opts, 'cls_loss_weight', 0.0))
        self.class_names = list(class_names) if class_names is not None else None
        self.category_to_idx = None
        self.ce_loss = None
        self.cls_head = None

        if self.cls_loss_weight > 0.0:
            if not self.class_names:
                raise ValueError(
                    "cls_loss_weight > 0 requires class_names from the training dataset (no hard-coded classes)."
                )

            self.category_to_idx = {c: i for i, c in enumerate(self.class_names)}
            nclass = len(self.class_names)
            self.opts.nclass = nclass

            embed_dim = getattr(getattr(self.clip, 'visual', None), 'output_dim', None)
            if embed_dim is None:
                raise ValueError("Could not infer CLIP image embedding dimension (visual.output_dim missing).")

            self.ce_loss = nn.CrossEntropyLoss()
            self.cls_head = nn.Linear(int(embed_dim), int(nclass))

        self.best_metric = -1e3

    def configure_optimizers(self):
        param_groups = [
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr},
        ]
        if self.cls_head is not None:
            param_groups.append({'params': self.cls_head.parameters(), 'lr': self.opts.linear_lr})

        optimizer = torch.optim.Adam(param_groups)
        return optimizer

    def _categories_to_label_idx(self, categories):
        if self.category_to_idx is None:
            return None
        # categories is typically a list[str] after default collate.
        idx = [self.category_to_idx[c] for c in categories]
        return torch.as_tensor(idx, device=self.device, dtype=torch.long)

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        loss = triplet_loss

        if self.cls_loss_weight > 0.0 and self.cls_head is not None:
            label_idx = self._categories_to_label_idx(category)
            logits_sk = self.cls_head(sk_feat.float())
            logits_img = self.cls_head(img_feat.float())
            cls_loss = 0.5 * (self.ce_loss(logits_sk, label_idx) + self.ce_loss(logits_img, label_idx))
            loss = loss + self.cls_loss_weight * cls_loss
            self.log('train_cls_loss', cls_loss, prog_bar=True)

        self.log('train_triplet_loss', triplet_loss)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss)
        return sk_feat, img_feat, category

    def validation_epoch_end(self, val_step_outputs):
        Len = len(val_step_outputs)
        if Len == 0:
            return
        query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)])
        gallery_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)])
        all_category = np.array(sum([list(val_step_outputs[i][2]) for i in range(Len)], []))


        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == category)] = True
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        
        mAP = torch.mean(ap)
        self.log('mAP', mAP)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        print ('mAP: {}, Best mAP: {}'.format(mAP.item(), self.best_metric))
