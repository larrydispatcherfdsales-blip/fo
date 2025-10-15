import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

print("Starting TTS model training process...")

# --- Paths ---
audio_file_path = "training_voice.mp3"
output_path = os.path.join(os.getcwd(), "trained_model_output")
os.makedirs(output_path, exist_ok=True)
print(f"Output directory created at: {output_path}")

try:
    # --- Step 1: Base model ka path dhoondna ---
    # Coqui ka Docker image model ko pehle se download karke rakhta hai
    manager = ModelManager()
    model_path = manager.get_model_path("tts_models/multilingual/multi-dataset/xtts_v2")
    print(f"Found base model at: {model_path}")

    # --- Step 2: Config aur Model ko load karna ---
    print("Loading base XTTS v2 model and config...")
    config = load_config(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
    print("Base model and weights loaded successfully.")

    # --- Step 3: Training arguments set karna ---
    print("Setting up training arguments...")
    # Hum batch size ko 1 kar rahe hain taake kam resources mein bhi chal jaye
    training_args = TrainerArgs(
        output_path=output_path,
        run_name="xtts_finetune_run",
        epochs=6, # Thore zyada epochs
        batch_size=1, 
        save_step=100,
        log_step=10,
        learning_rate=1e-5,
        use_ddp=False, # Distributed training band
        num_loader_workers=2,
        max_audio_len=262144
    )

    # --- Step 4: Trainer ko initialize karna ---
    print("Initializing the Trainer...")
    trainer = Trainer(
        args=training_args,
        config=config,
        output_path=output_path,
        model=model,
        train_samples_path=audio_file_path,
        eval_samples_path=audio_file_path
    )

    # --- Step 5: Training shuru karna ---
    print("Starting fine-tuning...")
    trainer.fit()
    print("Model training completed successfully!")
    print(f"Trained model saved in '{output_path}' directory.")

except Exception as e:
    print(f"An error occurred during training: {e}")
    import traceback
    traceback.print_exc()

