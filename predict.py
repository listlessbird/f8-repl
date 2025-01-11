import os
import gc
import torch
from cog import BasePredictor, Input, Path
from f5_tts.api import F5TTS

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        print("Loading models...")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.8)
            torch.cuda.empty_cache()

        # Verify model files exist
        required_files = [
            "models/F5-TTS/F5TTS_Base/model_1200000.safetensors",
            "models/F5-TTS/F5TTS_Base/vocab.txt",
            "models/vocos/config.yaml",
            "models/vocos/pytorch_model.bin"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required model file not found: {file_path}")
        
        self.model = F5TTS(
            model_type="F5-TTS",
            ckpt_file="models/F5-TTS/F5TTS_Base/model_1200000.safetensors",  
            vocab_file="models/F5-TTS/F5TTS_Base/vocab.txt",
            vocoder_name="vocos",
            use_ema=True,
            local_path="models/vocos",  # Local vocoder path
            device=self.device,
            ode_method="euler"
        )
        
        print("Models loaded successfully")

    def predict(
        self,
        ref_audio: Path = Input(description="Reference audio file"),
        ref_text: str = Input(description="Reference text (leave empty for auto-transcription)", default=""),
        text_to_generate: str = Input(description="Text to synthesize"),
        remove_silence: bool = Input(description="Remove silence from output", default=True),
        speed: float = Input(description="Speech speed (0.5-2.0)", default=1.0, ge=0.5, le=2.0),
        seed: int = Input(description="Random seed (-1 for random)", default=-1, ge=-1),
        cfg_strength: float = Input(description="Classifier-free guidance strength", default=2.0, ge=0.0),
        nfe_step: int = Input(description="Number of function evaluations", default=32, ge=4, le=64),
    ) -> Path:
        """Run TTS inference"""
        try:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            output_path = Path("output.wav")
            spec_path = Path("output_spec.png") 
            
            wav, sr, spect = self.model.infer(
                ref_file=str(ref_audio),
                ref_text=ref_text,
                gen_text=text_to_generate,
                file_wave=str(output_path),
                file_spect=str(spec_path),
                remove_silence=remove_silence,
                speed=speed,
                seed=seed,
                cfg_strength=cfg_strength,
                nfe_step=nfe_step
            )
            
            print(f"Generation completed successfully with seed {self.model.seed}")
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return output_path
            
        except Exception as e:
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise e

    def cleanup(self):
        """Cleanup any resources"""
        if hasattr(self, "device") and self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()