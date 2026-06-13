import json
import random
from datasets import load_dataset
import pandas as pd

def prepare_dataset():
    print("Loading HotpotQA dataset...")
    # Load train split of distractor setting because validation only has 'hard'
    dataset = load_dataset("hotpot_qa", "distractor", split="train")
    
    # Convert to pandas for easier filtering
    df = pd.DataFrame(dataset)
    
    # We want 25 of each: bridge/easy, bridge/hard, comparison/easy, comparison/hard
    sampled_dfs = []
    
    for q_type in ['bridge', 'comparison']:
        for q_level in ['easy', 'hard']:
            subset = df[(df['type'] == q_type) & (df['level'] == q_level)]
            
            # Sample 25
            if len(subset) >= 25:
                sampled = subset.sample(n=25, random_state=42)
            else:
                print(f"Warning: Only {len(subset)} questions found for {q_type}/{q_level}. Taking all.")
                sampled = subset
                
            sampled_dfs.append(sampled)
            
    final_df = pd.concat(sampled_dfs)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the dataset
    
    # Convert back to dict records
    records = final_df.to_dict('records')
    
    # Save to data/sampled_100_questions.json
    output_path = "data/sampled_100_questions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(records)} questions to {output_path}")

if __name__ == "__main__":
    prepare_dataset()
