"""Local asset loading for the official Reason1 processor (methods unchanged)."""
from transformers import AutoProcessor
from cosmos_predict2._src.reason1.tokenizer.processor import Processor

class LocalProcessor(Processor):
    def __init__(self, path):
        self.name='Qwen/Qwen2.5-VL-7B-Instruct'
        self.is_vision_tokenizer=True
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True)
        self.image_token_id=self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)
        self.video_token_id=self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)
        self.eos_id=self.processor.tokenizer.eos_token_id
        self.pad_id=self.processor.tokenizer.pad_token_id
