import os
from TTS.api import TTS

print("--- Starting TTS Model Training ---")

# Paths
audio_file_path = "training_voice.mp3"
output_path = os.path.join(os.getcwd(), "trained_model_output")

# Make sure the output directory exists
os.makedirs(output_path, exist_ok=True)
print(f"Output directory is: {output_path}")

try:
    # Load the base model
    # The COQUI_TOS_AGREED=1 in the workflow will handle the license
    print("Loading base XTTS v2 model...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    print("Base model loaded successfully.")

    # Start fine-tuning
    print(f"Starting fine-tuning on '{audio_file_path}'...")
    tts.finetune(
        audio_path=audio_file_path,
        model_name="custom_voice.pth",
        output_path=output_path,
        epochs=5,  # Let's try 5 epochs
        batch_size=2,
        num_loader_workers=2,
        max_audio_len=262144
    )
    print("--- MODEL TRAINING COMPLETED SUCCESSFULLY! ---")

except Exception as e:
    print(f"!!! AN ERROR OCCURRED DURING TRAINING: {e} !!!")
    import traceback
    traceback.print_exc()

