import os
from lunsft.lunslib import LunsDataset, LunsformterModel, ReadableResponseLayer

def main():
    # Path to your dataset
    dataset_path = 'examples/chat_data.txt'

    # Load dataset (plain text example, or adapt for CSV/JSON + options param)
    print("Loading dataset...")
    dataset = LunsDataset(dataset_path)

    # Initialize readable response heuristic layer (optional but improves quality)
    readable_layer = ReadableResponseLayer(max_attempts=5)

    # Create model
    model = LunsformterModel(dataset,
                             seq_len=20,
                             dim=64,
                             hidden_dim=128,
                             num_layers=2,
                             lr=0.02,
                             readable_layer=readable_layer)

    # Train
    print("Training model...")
    model.fit(epochs=100, verbose=True)

    # Save model
    save_prefix = 'sample_model'
    print(f"Saving model as '{save_prefix}' ...")
    model.save(save_prefix)

    # Load model back (just to show how)
    print(f"Loading model from '{save_prefix}' ...")
    loaded_model = LunsformterModel.load(save_prefix)

    # Test generation
    prompt = "Hello"
    print("\nPrompt:", prompt)
    output = loaded_model.generate(prompt, max_tokens=20, temperature=0.9)
    print("Generated:", output)

if __name__ == "__main__":
    main()