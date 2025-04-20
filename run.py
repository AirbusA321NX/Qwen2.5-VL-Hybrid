from transformers import AutoProcessor
from models.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

model_dir = "./model_checkpoint"
config = Qwen2_5_VLConfig(base_model_path=model_dir)
processor = AutoProcessor.from_pretrained(model_dir)
model = Qwen2_5_VLForConditionalGeneration(config)

print("✅ Model loaded successfully")
