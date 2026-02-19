from datasets import load_dataset

def inspect():
    with open("dataset_inspection.txt", "w", encoding="utf-8") as f:
        try:
            dataset = load_dataset("teyler/epstein-files-20k", split="train", streaming=True)
            f.write(f"Dataset Features: {dataset.features}\n")
            
            f.write("\n--- First 5 Examples ---\n")
            for i, item in enumerate(dataset.take(5)):
                f.write(f"\nItem {i}:\n")
                f.write(f"Keys: {list(item.keys())}\n")
                f.write(f"Text Preview: {item.get('text', '')[:200]}...\n")
                if 'meta' in item:
                    f.write(f"Meta: {item['meta']}\n")
        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    inspect()
