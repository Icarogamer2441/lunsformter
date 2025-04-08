from lunsft.lunslib import LunsDataset, LunsformterModel

# === INITIAL TRAINING ===
dataset_path = 'examples/chat_data.txt'
dataset = LunsDataset(dataset_path, num_merges=500)
model = LunsformterModel(dataset, seq_len=15, dim=120, hidden_dim=256, num_layers=4, chunk_size=8)

print("Starting initial training...")
model.fit(epochs=300)
print("Initial training done.\n")

prompt = "Hello my name is [NAME] and"
print(f"\nPrompt before saving: {prompt}")

print("\n=== Generation before saving (temperature=1.0) ===")
gen1 = model.generate(prompt, max_tokens=12, temperature=1.0, verbose=False)
print(f"Generated:\n{gen1}")

save_prefix = 'examples/chat_model'
print("\nSaving model to disk...")
model.save(save_prefix)
print(f"Model saved as files with prefix '{save_prefix}'\n")

print("Loading the saved model back from disk...")
loaded_model = LunsformterModel.load(save_prefix)
print("Model loaded successfully.\n")

print("\n=== Generation from loaded model (temperature=0.8, verbose mode) ===")
gen2 = loaded_model.generate(prompt, max_tokens=12, temperature=0.8, verbose=True)
print(f"Generated after reload:\n{gen2}")

print("\n=== Fine-tuning the loaded model for 200 more epochs... ===")
loaded_model.fine_tune(dataset, epochs=200)
print("Fine-tuning complete.\n")

print("\n=== Generation after fine-tuning (temperature=1.2) ===")
gen3 = loaded_model.generate(prompt, max_tokens=12, temperature=1.2, verbose=False)
print(f"Generated after fine-tune:\n{gen3}")