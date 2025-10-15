import os
from TTS.api import TTS

print("--- Starting TTS Model Training (Final & Simplified) ---")

# Paths
audio_file_path = "training_voice.mp3"
output_path = os.path.join(os.getcwd(), "trained_model_output")

# Make sure the output directory exists
os.makedirs(output_path, exist_ok=True)
print(f"Output directory is: {output_path}")

try:
    # --- YAHAN TABDEELI KI GAYI HAI ---
    # Hum direct base model ko load kar rahe hain, ModelManager ke baghair.
    # Yeh tareeqa TTS v0.22.0 ke liye bilkul sahi hai.
    print("Loading base XTTS v2 model directly...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("Base model loaded successfully.")

    # --- Fine-tuning ka process shuru karna ---
    print(f"Starting fine-tuning with '{audio_file_path}'...")
    tts.finetune(
        audio_path=audio_file_path,
        model_name="custom_voice.pth", # Trained model ka naam
        output_path=output_path,
        epochs=10,  # Hum 10 epochs try karenge
        batch_size=2,
        num_loader_workers=2,
        max_audio_len=262144  # Audio ki max length
    )
    print("--- MODEL TRAINING COMPLETED SUCCESSFULLY! ---")

except Exception as e:
    print(f"!!! AN ERROR OCCURRED DURING TRAINING: {e} !!!")
    import traceback
    traceback.print_exc()

