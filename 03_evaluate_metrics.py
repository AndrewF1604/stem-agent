import json
import re
import string
from collections import Counter
import ollama
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm

load_dotenv()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def llm_as_judge(question, prediction, ground_truth):
    prompt = f"""You are an expert evaluator for question answering tasks.
Your job is to determine if the model's prediction is factually equivalent to the ground truth answer.
Tolerate variations in wording, synonyms, and extra helpful context as long as the core facts match.
Do not penalize formatting differences.

Question: {question}
Ground Truth: {ground_truth}
Model Prediction: {prediction}

Is the model prediction correct? Answer ONLY with 'YES' or 'NO'."""
    
    try:
        response = ollama.chat(
            model='gemma2',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        answer = response['message']['content'].strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"Ollama error in evaluator: {e}")
        return False

def main():
    if os.path.exists("data/evaluated_responses.json"):
        with open("data/evaluated_responses.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print("Loaded existing evaluated_responses.json to resume progress.")
    else:
        try:
            with open("data/raw_responses.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("data/raw_responses.json not found.")
            return

    for item in tqdm(data, desc="Evaluating"):
        question = item['question']
        truth = item['true_answer']
        
        if 'metrics' not in item:
            item['metrics'] = {}
            
        for config, responses in item['responses'].items():
            if config in item['metrics'] and len(item['metrics'][config]) == len(responses):
                continue # Already fully evaluated this config
                
            item['metrics'][config] = []
            for resp in responses:
                em = int(exact_match_score(resp, truth))
                f1 = f1_score(resp, truth)
                judge = int(llm_as_judge(question, resp, truth))
                
                item['metrics'][config].append({
                    'em': em,
                    'f1': f1,
                    'judge': judge
                })
                
        # Incremental save
        with open("data/evaluated_responses.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    print("Evaluation complete. Saved to data/evaluated_responses.json")

if __name__ == "__main__":
    main()
