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

        self.triplet_loss_weight = float(getattr(self.opts, 'triplet_loss_weight', 1.0))

        # Contrastive loss (InfoNCE) on sketch/image features.
        self.contrastive_loss_weight = float(getattr(self.opts, 'contrastive_loss_weight', 0.0))
        self.contrastive_temperature = float(getattr(self.opts, 'contrastive_temperature', 0.07))

        # Loss balancing to avoid overpowering due to magnitude differences.
        self.loss_balance = str(getattr(self.opts, 'loss_balance', 'fixed'))
        self.loss_ema_beta = float(getattr(self.opts, 'loss_ema_beta', 0.98))
        self.loss_warmup_steps = int(getattr(self.opts, 'loss_warmup_steps', 0))
        self.register_buffer('ema_triplet', torch.tensor(1.0))
        self.register_buffer('ema_contrastive', torch.tensor(1.0))
        self.register_buffer('ema_cls', torch.tensor(1.0))

        # Negative mining strategy for triplet loss.
        self.negative_mining = str(getattr(self.opts, 'negative_mining', 'random'))

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

    def on_load_checkpoint(self, checkpoint) -> None:
        """Backwards-compatible checkpoint loading.

        Older checkpoints won't have newly introduced EMA buffers; initialize them
        so strict state_dict loading doesn't fail.
        """
        state_dict = checkpoint.get('state_dict', None)
        if not isinstance(state_dict, dict):
            return
        for key in ('ema_triplet', 'ema_contrastive', 'ema_cls'):
            if key not in state_dict:
                state_dict[key] = torch.tensor(1.0)

    def _warmup_scale(self) -> float:
        steps = int(self.loss_warmup_steps)
        if steps <= 0:
            return 1.0
        # In Lightning, global_step is the number of optimizer steps taken so far.
        return float(min(1.0, (self.global_step + 1) / float(steps)))

    def _ema_update(self, name: str, value: torch.Tensor) -> None:
        # Track EMA on detached scalar magnitude.
        v = value.detach().float()
        if v.numel() != 1:
            v = v.mean()
        beta = float(self.loss_ema_beta)
        beta = 0.0 if beta < 0.0 else (0.9999 if beta > 0.9999 else beta)

        if name == 'triplet':
            self.ema_triplet.mul_(beta).add_(v * (1.0 - beta))
        elif name == 'contrastive':
            self.ema_contrastive.mul_(beta).add_(v * (1.0 - beta))
        elif name == 'cls':
            self.ema_cls.mul_(beta).add_(v * (1.0 - beta))

    def _normalize_by_ema(self, name: str, value: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        if name == 'triplet':
            denom = self.ema_triplet.clamp_min(eps)
        elif name == 'contrastive':
            denom = self.ema_contrastive.clamp_min(eps)
        elif name == 'cls':
            denom = self.ema_cls.clamp_min(eps)
        else:
            denom = torch.tensor(1.0, device=value.device)
        return value / denom

    def _contrastive_loss(self, sk_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE loss on in-batch positives (diagonal) and negatives (off-diagonal)."""
        if sk_feat.ndim != 2 or img_feat.ndim != 2:
            raise ValueError('Expected 2D feature tensors [B, D].')
        if sk_feat.shape[0] != img_feat.shape[0]:
            raise ValueError('sk_feat and img_feat must have the same batch size.')

        temperature = max(float(self.contrastive_temperature), 1e-6)
        sk_norm = F.normalize(sk_feat.float(), dim=-1)
        img_norm = F.normalize(img_feat.float(), dim=-1)

        logits = (sk_norm @ img_norm.t()) / temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss_s2i = F.cross_entropy(logits, targets)
        loss_i2s = F.cross_entropy(logits.t(), targets)
        return 0.5 * (loss_s2i + loss_i2s)

    def _mine_batch_hard_negative_img_feats(
        self,
        sk_feat: torch.Tensor,
        img_feat: torch.Tensor,
        categories,
    ):
        """Mine hardest negatives from in-batch images with different category.

        Returns mined negative features with shape [B, D], or None if mining is not possible.
        """
        if categories is None:
            return None
        if not isinstance(categories, (list, tuple)):
            return None
        batch_size = int(sk_feat.shape[0])
        if batch_size < 2:
            return None
        if len(categories) != batch_size:
            return None

        # Pairwise distances between each sketch and each image in the batch.
        sk_norm = F.normalize(sk_feat.float(), dim=-1)
        img_norm = F.normalize(img_feat.float(), dim=-1)
        dist = 1.0 - (sk_norm @ img_norm.t())  # [B, B]

        mined_indices = []
        for i, cat in enumerate(categories):
            neg_js = [j for j, c in enumerate(categories) if c != cat]
            if not neg_js:
                mined_indices.append(None)
                continue
            neg_js_t = torch.as_tensor(neg_js, device=dist.device, dtype=torch.long)
            j_rel = int(torch.argmin(dist[i, neg_js_t]).item())
            mined_indices.append(int(neg_js[j_rel]))

        if any(j is None for j in mined_indices):
            return None
        idx = torch.as_tensor(mined_indices, device=img_feat.device, dtype=torch.long)
        return img_feat.index_select(0, idx)

    @staticmethod
    def _ap_at_k(scores: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
        """Average Precision at K for a single query.

        scores: higher is better, shape [N]
        target: bool or {0,1}, shape [N]
        """
        n = int(scores.numel())
        if n == 0:
            return scores.new_tensor(0.0)
        k_eff = min(int(k), n)
        topk_idx = torch.topk(scores, k=k_eff, largest=True).indices
        rel = target[topk_idx].to(dtype=torch.float32)

        # precision at each rank
        cumsum = torch.cumsum(rel, dim=0)
        ranks = torch.arange(1, k_eff + 1, device=scores.device, dtype=torch.float32)
        precision = cumsum / ranks

        # AP@K = sum(precision@i * rel_i) / min(num_relevant_total, K)
        num_rel_total = target.to(dtype=torch.int64).sum().to(dtype=torch.float32)
        denom = torch.clamp(num_rel_total, max=float(k_eff)).clamp_min(1.0)
        ap = (precision * rel).sum() / denom
        return ap

    @staticmethod
    def _precision_at_k(scores: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
        """Precision at K for a single query."""
        n = int(scores.numel())
        if n == 0:
            return scores.new_tensor(0.0)
        k_eff = min(int(k), n)
        topk_idx = torch.topk(scores, k=k_eff, largest=True).indices
        rel = target[topk_idx].to(dtype=torch.float32)
        return rel.sum() / float(k_eff)

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

        neg_feat = None
        if self.negative_mining == 'batch_hard':
            neg_feat = self._mine_batch_hard_negative_img_feats(sk_feat, img_feat, category)

        # Fallback to dataset-sampled negatives.
        if neg_feat is None:
            neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self._ema_update('triplet', triplet_loss)

        contrastive_loss = None
        if self.contrastive_loss_weight > 0.0:
            contrastive_loss = self._contrastive_loss(sk_feat, img_feat)
            self._ema_update('contrastive', contrastive_loss)
            self.log('train_contrastive_loss', contrastive_loss, prog_bar=True)

        cls_loss = None
        if self.cls_loss_weight > 0.0 and self.cls_head is not None:
            label_idx = self._categories_to_label_idx(category)
            logits_sk = self.cls_head(sk_feat.float())
            logits_img = self.cls_head(img_feat.float())
            cls_loss = 0.5 * (self.ce_loss(logits_sk, label_idx) + self.ce_loss(logits_img, label_idx))
            self._ema_update('cls', cls_loss)
            self.log('train_cls_loss', cls_loss, prog_bar=True)

        # ---- Total loss composition ----
        warm = self._warmup_scale()

        # Always keep triplet as the anchor objective.
        w_triplet = float(self.triplet_loss_weight)
        w_contrastive = float(self.contrastive_loss_weight) * warm
        w_cls = float(self.cls_loss_weight) * warm

        if self.loss_balance == 'ema_normalize':
            total = w_triplet * self._normalize_by_ema('triplet', triplet_loss)
            if contrastive_loss is not None:
                total = total + w_contrastive * self._normalize_by_ema('contrastive', contrastive_loss)
            if cls_loss is not None:
                total = total + w_cls * self._normalize_by_ema('cls', cls_loss)
        else:
            total = w_triplet * triplet_loss
            if contrastive_loss is not None:
                total = total + w_contrastive * contrastive_loss
            if cls_loss is not None:
                total = total + w_cls * cls_loss

        self.log('loss_warmup_scale', warm, prog_bar=False)
        self.log('ema_triplet', self.ema_triplet, prog_bar=False)
        self.log('ema_contrastive', self.ema_contrastive, prog_bar=False)
        self.log('ema_cls', self.ema_cls, prog_bar=False)

        self.log('train_triplet_loss', triplet_loss)
        self.log('train_loss', total, prog_bar=True)
        return total

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


        ## Retrieval metrics (category-level SBIR)
        # Default evaluation cutoff.
        k = 200
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all), device=query_feat_all.device)
        prec = torch.zeros(len(query_feat_all), device=query_feat_all.device)
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            scores = -1 * self.distance_fn(sk_feat.unsqueeze(0), gallery).squeeze(0)
            target = torch.zeros(len(gallery), dtype=torch.bool, device=query_feat_all.device)
            target[np.where(all_category == category)] = True
            ap[idx] = self._ap_at_k(scores, target, k=k)
            prec[idx] = self._precision_at_k(scores, target, k=k)
        
        mAP_200 = torch.mean(ap)
        p_200 = torch.mean(prec)

        # Log new metrics; keep 'mAP' as an alias for backwards compatibility.
        self.log('mAP@200', mAP_200)
        self.log('P@200', p_200)
        self.log('mAP', mAP_200)
        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP_200.item()) else mAP_200.item()
        print('mAP@200: {}, P@200: {}, Best mAP@200: {}'.format(mAP_200.item(), p_200.item(), self.best_metric))
