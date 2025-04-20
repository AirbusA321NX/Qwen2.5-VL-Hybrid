from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch import nn
import torch


class Qwen2_5_VLForConditionalGeneration(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        from transformers import AutoModel, AutoTokenizer
        self.vision_encoder = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_decoder = AutoModel.from_pretrained("Qwen/Qwen-1_8B-Chat")

        self.projection = nn.Linear(self.vision_encoder.config.hidden_size, self.text_decoder.config.hidden_size)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        image_embeds = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        projected = self.projection(image_embeds)

        decoder_input_ids = input_ids
        output = self.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask
        )

        logits = output.last_hidden_state

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            decoder_hidden_states=output.hidden_states,
            decoder_attentions=output.attentions,
        )
