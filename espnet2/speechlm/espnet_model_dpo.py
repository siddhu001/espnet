#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from typing import Dict, Mapping, Tuple

import torch
from typeguard import typechecked
from copy import deepcopy

from espnet2.speechlm.core_lm.abs_core_lm import AbsCoreLM
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetSpeechLMDPOModel(AbsESPnetModel):

    @typechecked
    def __init__(
        self,
        corelm: AbsCoreLM,
        criterion,
        beta: float = 0.1,
        sft_weight: float = 0.0,
        use_sft_loss_mask: bool = False,
        no_dpo_when_sft: bool = False,
        debug: bool = False,
        extract_feats_in_collect_stats: bool = False,
        mean_on_sft: bool = True,
    ):
        super().__init__()

        self.corelm = corelm
        self.criterion = criterion
        self.beta = beta
        self.sft_weight = sft_weight
        self.use_sft_loss_mask = use_sft_loss_mask
        self.no_dpo_when_sft = no_dpo_when_sft
        self.debug = debug
        self.mean_on_sft = mean_on_sft
        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

        # Currently, always use itself as the reference LM. Freeze it
        self.reflm = deepcopy(corelm)
        for p in self.reflm.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        dec_seq: torch.Tensor,
        loss_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        sft_loss_mask = kwargs.pop("sft_loss_mask", None)

        pos_dec_seq = dec_seq[..., 0]
        neg_dec_seq = dec_seq[..., 1]
        pos_loss_mask = loss_mask[..., 0]
        neg_loss_mask = loss_mask[..., 1]

        # [B, T, nq, 2] -> [2 * B, T, nq]
        dec_seq = torch.cat([pos_dec_seq, neg_dec_seq], dim=0)
        loss_mask = torch.cat([pos_loss_mask, neg_loss_mask], dim=0)

        # (1) LM forward
        policy_ce_loss, policy_elem_logp, _, _ = self.criterion(
            *self.corelm(dec_seq, loss_mask)
        )

        with torch.no_grad():
            ref_ce_loss, ref_elem_logp, _, _ = self.criterion(
                *self.reflm(dec_seq, loss_mask)
            )
        if sft_loss_mask is not None:
            if sft_loss_mask.dim() == 4:
                pos_sft_mask = sft_loss_mask[..., 0]
                neg_sft_mask = sft_loss_mask[..., 1]
            else:
                pos_sft_mask = sft_loss_mask
        if self.no_dpo_when_sft and pos_sft_mask is not None:
            if self.debug:
                import pdb;pdb.set_trace()
            pos_sft_mask = pos_sft_mask[:,:pos_loss_mask.shape[1],:]
            neg_sft_mask = neg_sft_mask[:,:neg_loss_mask.shape[1],:]
            pos_dpo_mask = pos_loss_mask.bool() & (~pos_sft_mask.bool())
            neg_dpo_mask = neg_loss_mask.bool() & (~neg_sft_mask.bool())
            dpo_loss_mask = torch.cat([pos_dpo_mask, neg_dpo_mask], dim=0)
            policy_logp = - (
                policy_elem_logp * dpo_loss_mask[:,1:,:].to(policy_elem_logp.dtype)
            ).sum(dim=(1, 2))
            ref_logp = - (ref_elem_logp * dpo_loss_mask[:,1:,:].to(ref_elem_logp.dtype)).sum(
                dim=(1, 2)
            )
        else:
            # (2) DPO
            # neg-log-likelihood -> log-likelihood
            policy_logp = - policy_elem_logp.sum(dim=(1, 2))
            ref_logp = - ref_elem_logp.sum(dim=(1, 2))

        num = len(policy_logp) // 2
        pos_policy_logp = policy_logp[:num]
        neg_policy_logp = policy_logp[num:]
        pos_ref_logp = ref_logp[:num]
        neg_ref_logp = ref_logp[num:]

        loss, stats = self.dpo(
            pos_policy_logp=pos_policy_logp,
            neg_policy_logp=neg_policy_logp,
            pos_ref_logp=pos_ref_logp,
            neg_ref_logp=neg_ref_logp,
        )

        # (3) optional supervised fine-tuning (SFT) objective on positive samples
        if self.sft_weight > 0:
            sft_elem_loss = policy_elem_logp[:num]
            if self.use_sft_loss_mask and sft_loss_mask is not None:
                # Accept either a per-sample mask [B, T, nq] or a paired mask
                # [B, T, nq, 2]; only the positive side contributes to SFT.
                sft_mask = pos_sft_mask.bool() & pos_loss_mask.bool()
            else:
                sft_mask = pos_loss_mask
            sft_weight = sft_mask.sum().float()
            if sft_weight > 0:
                # import pdb;pdb.set_trace()
                masked_loss = sft_elem_loss * sft_mask[:,1:,:].to(sft_elem_loss.dtype)
                if self.criterion.loss_type == "mean" and self.mean_on_sft:
                    sft_loss = masked_loss.sum() / sft_weight
                else:
                    sft_loss = masked_loss.sum()
            else:
                # zero gradient contribution if no positive samples are present
                sft_loss = torch.zeros_like(sft_elem_loss.sum())
            # import pdb;pdb.set_trace()
            if self.debug:
                import pdb;pdb.set_trace()
            loss = loss + self.sft_weight * sft_loss
            stats.update({
                "loss_sft": sft_loss.clone().detach(),
                "sft_weight": sft_weight.clone().detach(),
            })

        loss, stats, num = force_gatherable((loss, stats, num), loss.device)

        return loss, stats, num
    
    def dpo(
        self,
        pos_policy_logp: torch.Tensor,
        neg_policy_logp: torch.Tensor,
        pos_ref_logp: torch.Tensor,
        neg_ref_logp: torch.Tensor,
    ):
        logits = (pos_policy_logp - neg_policy_logp) - (pos_ref_logp - neg_ref_logp)
        loss = - torch.nn.functional.logsigmoid(logits * self.beta)

        pos_reward = pos_policy_logp - pos_ref_logp
        neg_reward = neg_policy_logp - neg_ref_logp
        reward_gap = pos_reward - neg_reward
        win_rate = (reward_gap > 0).float().sum() / len(reward_gap)
        loss_rate = (reward_gap < 0).float().sum() / len(reward_gap)
        equal_rate = (reward_gap == 0).float().sum() / len(reward_gap)

        stats = {
            "loss_dpo": loss,
            "pos_reward": pos_reward,
            "neg_reward": neg_reward,
            "reward_gap": reward_gap,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "equal_rate": equal_rate,
        }
        stats = {k: v.clone().detach().mean() for k, v in stats.items()}

        return loss.mean(), stats
    
    def collect_feats(self, **kwargs):
        raise NotImplementedError
    
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """ Make the reflm the same as corelm """
        keys = [key for key in state_dict if key.startswith('corelm')]
        for key in keys:
            ref_key = key.replace("corelm.", "reflm.")
            state_dict[ref_key] = state_dict[key]
        super().load_state_dict(state_dict, strict, assign)
    
    def state_dict(self, **kwargs):
        """ Only save the corelm parameters, not reflm """
        state_dict = super().state_dict(**kwargs)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("reflm.")}
        return state_dict
