from lunsft.lunslib import LunsDataset, LunsformterModel

dataset_path = 'examples/chat_data.txt'
print("Loading dataset from:", dataset_path)
dataset = LunsDataset(dataset_path, num_merges=200)

print("Initializing model...")
model = LunsformterModel(dataset, seq_len=12, dim=64, hidden_dim=128, num_layers=2, chunk_size=4)

print("Training the model (500 epochs)...")
model.fit(epochs=500)

prompt = "Hello my name is"
print("\nPrompt:", prompt)
print("\nGenerating after initial training:")
print(model.generate(prompt, max_tokens=12, temperature=1.0, verbose=False))

save_prefix = 'examples/test_run_model'
print(f"\nSaving model to '{save_prefix}'...")
model.save(save_prefix)

print("\nLoading saved model back...")
loaded_model = LunsformterModel.load(save_prefix)

print("\nGenerating from the loaded model:")
print(loaded_model.generate(prompt, max_tokens=12, temperature=0.8, verbose=True))

print("\nFine-tuning loaded model on same dataset (100 epochs)...")
loaded_model.fine_tune(dataset, epochs=100)

print("\nGenerating after fine-tuning:")
print(loaded_model.generate(prompt, max_tokens=12, temperature=1.2, verbose=False))

print("\n=== TEST COMPLETED SUCCESSFULLY ===")