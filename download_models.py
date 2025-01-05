import os
from huggingface_hub import snapshot_download

def download_models():
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download F5-TTS model and vocab
    print("Downloading F5-TTS model...")
    snapshot_download(
        repo_id="SWivid/F5-TTS",
        allow_patterns=["F5TTS_Base/model_1200000.safetensors", "F5TTS_Base/vocab.txt"],
        local_dir="./models/F5-TTS"
    )
    
    # Download Vocos vocoder
    print("Downloading Vocos vocoder...")
    snapshot_download(
        repo_id="charactr/vocos-mel-24khz",
        local_dir="./models/vocos"
    )
    
    # Download Whisper model
    print("Downloading Whisper model...")
    from faster_whisper import download_model
    download_model("large-v3", output_dir="./models/whisper")

if __name__ == "__main__":
    download_models()
    