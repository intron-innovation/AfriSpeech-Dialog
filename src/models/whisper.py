import torch
from src.models.models import Model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Whisper(Model):
    def __init__(self, pretrained_model_path, **kwargs):
        super().__init__(pretrained_model_path, **kwargs)
        
        # Determine the device and dtype
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load the model
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_path, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)

        # Load the processor
        self.processor = AutoProcessor.from_pretrained(pretrained_model_path)

        # Set up the pipeline. This does chunked long-form transcription
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30, #remove chunk_length_s for sequential long form transcription
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe(self, audio_files: list, batch_size: int = 2) -> list:
        results = self.pipe(audio_files, batch_size=batch_size)
        transcripts = [result['text'] for result in results]
        return transcripts