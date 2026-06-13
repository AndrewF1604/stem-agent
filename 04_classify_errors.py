import json
from collections import Counter
import ollama
from dotenv import load_dotenv
import os
from tqdm import tqdm

load_dotenv()

def classify_error(question, context, ground_truth, prediction):
    prompt = f"""You are an expert linguistic error annotator.
The model provided an INCORRECT answer to a multi-hop question.
Classify the error into EXACTLY ONE of the following 5 categories:

1. Hallucination - The answer contains information not present in the provided texts.
2. Multi-hop reasoning error - The model found the first hop correctly but failed the second hop (wrong conclusion).
3. Format error - The model knows the answer but provided it in an incorrect format (e.g. full sentence instead of entity).
4. Negation error - The model missed a "not" or similar negation in the question or text.
5. Entity confusion - The model chose a related but incorrect entity from the text (e.g. wrong person, wrong city).

Question: {question}
Texts: {context}
Ground Truth: {ground_truth}
Model Prediction: {prediction}

Respond ONLY with the number (1-5) and the category name, e.g. "5. Entity confusion"."""

    try:
        response = ollama.chat(
            model='gemma2',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        
        answer = response['message']['content'].strip()
        if "1" in answer or "Hallucination" in answer: return "Hallucination"
        if "2" in answer or "reasoning" in answer: return "Multi-hop reasoning error"
        if "3" in answer or "Format" in answer: return "Format error"
        if "4" in answer or "Negation" in answer: return "Negation error"
        if "5" in answer or "Entity" in answer: return "Entity confusion"
        
        return "Unknown error"
    except Exception as e:
        print(f"Ollama error: {e}")
        return "API Error"

def analyze_stochasticity(responses, metrics_list):
    wrong_answers = []
    for resp, metric in zip(responses, metrics_list):
        if metric['judge'] == 0:
            wrong_answers.append(resp)
            
    if len(wrong_answers) == 0:
        return "No Error"
        
    if len(wrong_answers) == len(responses):
        unique_wrong = set(wrong_answers)
        if len(unique_wrong) == 1:
            return "Systematic"
            
    return "Stochastic"

def format_context(context_data):
    texts = []
    for title, sentences in zip(context_data['title'], context_data['sentences']):
        text = " ".join(sentences)
        texts.append(f"Title: {title}\nText: {text}")
    return "\n\n".join(texts)

def main():
    if os.path.exists("data/classified_errors.json"):
        with open("data/classified_errors.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print("Loaded existing classified_errors.json to resume progress.")
    else:
        try:
            with open("data/evaluated_responses.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("data/evaluated_responses.json not found.")
            return
        
    try:
        with open("data/sampled_100_questions.json", "r", encoding="utf-8") as f:
            questions = {q['id']: q for q in json.load(f)}
    except FileNotFoundError:
        print("data/sampled_100_questions.json not found.")
        return

    for item in tqdm(data, desc="Classifying errors"):
        if 'errors' not in item:
            item['errors'] = {}
            
        q_id = item['id']
        context = format_context(questions[q_id]['context'])
        
        for config, responses in item['responses'].items():
            metrics = item['metrics'][config]
            
            if config in item['errors'] and len(item['errors'][config]['classifications']) == len(responses):
                continue # Already classified
                
            item['errors'][config] = {
                'stochasticity': analyze_stochasticity(responses, metrics),
                'classifications': []
            }
            
            for resp, metric in zip(responses, metrics):
                if metric['judge'] == 0:
                    error_type = classify_error(item['question'], context, item['true_answer'], resp)
                    item['errors'][config]['classifications'].append(error_type)
                else:
                    item['errors'][config]['classifications'].append("Correct")
                    
        # Incremental save
        with open("data/classified_errors.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    print("Classification complete. Saved to data/classified_errors.json")

if __name__ == "__main__":
    main()
