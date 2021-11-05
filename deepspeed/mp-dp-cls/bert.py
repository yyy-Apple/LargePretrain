import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaConfig, RobertaForSequenceClassification
import loguru
from mpu import get_model_parallel_world_size

logger = loguru.logger
class ShardedRoBERTa(nn.Module):
    def __init__(self, args):
        super(ShardedRoBERTa, self).__init__()
        self.args = args
        self.num_labels = args.num_labels
        config = RobertaConfig.from_pretrained(args.checkpoint, num_labels=self.num_labels)
        self.model = RobertaForSequenceClassification.from_pretrained(args.checkpoint, from_tf=False, config=config)
        self.loss_fct = CrossEntropyLoss()

    def shard(self):
        # One node has 4 GPUs in our cluster, so we shard the model into 4 GPUs
        assert torch.cuda.is_available(), "CUDA not available"
        assert torch.cuda.device_count() == 4, "Not all 4 GPUs are available"

        world_size = get_model_parallel_world_size()

        # Move embedding layers to cuda:0
        self.model.roberta.embeddings.to("cuda:0")

        # We shard the transformer layers evenly into 4 devices
        layer_num = len(self.model.roberta.encoder.layer)
        assert layer_num % world_size == 0
        layer_per_device = layer_num / world_size

        for model_rank in [0, 1, 2, 3]:
            layer_start_idx = int(model_rank * layer_per_device)  # e.g. 0
            layer_end_idx = int(layer_start_idx + layer_per_device)  # e.g. 6
            for i in range(layer_start_idx, layer_end_idx):
                self.model.roberta.encoder.layer[i].to(f"cuda:{model_rank}")

        # Move classifier layers to cuda:3
        self.model.classifier.to("cuda:3")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
    ):

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)
        if token_type_ids is None:
            if hasattr(self.model.roberta.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.model.roberta.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.model.roberta.get_extended_attention_mask(
            attention_mask,
            input_shape,
            device
        )

        embedding_output = self.model.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        sequence_output = forward_encoder(
            self=self.model.roberta.encoder,
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask
        )

        logits = self.model.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # Move device so we can access from GPU:0
        loss = loss.to("cuda:0")
        logits = logits.to("cuda:0")
        return loss, logits

    @property
    def encoder(self):
        return self.model.roberta.encoder


def forward_encoder(
        self,  # Here self is self.encoder
        hidden_states,
        attention_mask=None,
):
    for i, layer_module in enumerate(self.layer):
        layer_outputs = layer_module(
            hidden_states.to(layer_module.output.dense.weight.device),
            attention_mask.to(layer_module.output.dense.weight.device),
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
        )
        hidden_states = layer_outputs[0]
    return hidden_states

"""
vim /home/pliu3/DeepSpeed/venv/lib/python3.8/site-packages/transformers/models/roberta/modeling_roberta.py
"""