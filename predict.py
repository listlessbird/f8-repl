from typing import Any
import torch
import os
from cog import BasePredictor, Input, Path
from faster_whisper import WhisperModel

from f5_tts.model import CFM, DiT
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    target_sample_rate,
)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        print("Loading models...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        
        self.asr_model = WhisperModel(
            model_size_or_path="large-v3",
            device=device,
            compute_type=compute_type,
            download_root="./models/whisper"
        )
        
        self.model_cls = DiT
        self.model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        
        self.ckpt_file = "models/F5-TTS/F5TTS_Base/model_1200000.safetensors"
        self.vocab_file = "models/F5-TTS/F5TTS_Base/vocab.txt"
        
        self.vocoder = load_vocoder(
            is_local=True,
            local_path="models/vocos"
        )
        self.model = load_model(
            self.model_cls,
            self.model_cfg,
            self.ckpt_file,
            vocab_file=self.vocab_file
        )
        
        print("Models loaded successfully")

    def predict(
        self,
        ref_audio: Path = Input(description="Reference audio file"),
        ref_text: str = Input(description="Reference text (leave empty for auto-transcription)", default=""),
        text_to_generate: str = Input(description="Text to synthesize"),
        remove_silence: bool = Input(description="Remove silence from output", default=True),
        speed: float = Input(description="Speech speed (0.5-2.0)", default=1.0, ge=0.5, le=2.0),
    ) -> Path:
        """Run TTS inference"""
        
        ref_audio_path = str(ref_audio)
        
        if not ref_text.strip():
            print("No reference text provided, transcription will be handled internally...")
            ref_text = ""
            
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

        with torch.inference_mode():
            final_wave, final_sample_rate, _ = infer_process(
                ref_audio,
                ref_text, 
                text_to_generate,
                self.model,
                self.vocoder,
                speed=speed
            )

        output_path = Path("output.wav")
        import soundfile as sf
        sf.write(str(output_path), final_wave, target_sample_rate)

        return output_path