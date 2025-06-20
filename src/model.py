import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from peft import LoraConfig, get_peft_model
from typing import Optional


class FSAConfig(PretrainedConfig):
    lora_config: LoraConfig
    backbone_id: str = "meta-llama/Llama-3.2-3B"
    num_labels = 3


class FSAModel(PreTrainedModel):
    """Financial Sentiment Analysis model with LlamaModel backbone"""

    config_class = FSAConfig

    def __init__(self, config: FSAConfig):
        super(FSAModel, self).__init__(config)

        self.llama = LlamaModel.from_pretrained(config.backbone_id, config)
        self.score = torch.nn.Linear(self.llama.config.hidden_size, config.num_labels, bias=False)
        
        for param in self.llama.parameters():
            param.requires_grad = False

        self.model = get_peft_model(self.llama, config.lora_config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> SequenceClassifierOutput:
        transformer_outputs: BaseModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits,
        )
