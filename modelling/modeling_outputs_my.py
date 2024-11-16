import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import Seq2SeqLMOutput

@dataclass
class Seq2SeqLMOutput_my(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_lastLayer_self_att: Optional[torch.FloatTensor] = None
    decoder_inputs_embeds_position: torch.FloatTensor = None
    decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    cross_attention_without_res: Optional[torch.FloatTensor] = None
    multi_lm_logit: torch.FloatTensor = None

@dataclass
class BaseModelOutputWithPastAndCrossAttentions_my(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    lastLayer_self_att: torch.FloatTensor = None
    inputs_embeds_position: torch.FloatTensor = None
    cross_attention_without_res: torch.FloatTensor = None

@dataclass
class Seq2SeqModelOutput_my(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    decoder_lastLayer_self_att : torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    decoder_inputs_embeds_position: torch.FloatTensor = None
    cross_attention_without_res: torch.FloatTensor = None