from lunsft import LunsDataset, LunsformterModel

dataset_path = 'examples/chat_data.txt'
dataset = LunsDataset(dataset_path)
model = LunsformterModel(dataset)

print("Starting training...")
model.fit(epochs=500)
print("Training complete.\n")

prompt = "Hello my name is [NAME] and"
print(f"Prompt: {prompt}")
generated = model.generate(prompt, max_tokens=12)
print(f"Generated:\n{generated}")