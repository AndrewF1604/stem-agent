import json
import os
import ollama
import wikipedia
import time
from tqdm import tqdm

def search_wikipedia(query: str) -> str:
    """Wyszukuje hasło w Wikipedii i zwraca podsumowanie pierwszego wyniku."""
    try:
        wikipedia.set_lang("en") # HotpotQA is in English
        results = wikipedia.search(query)
        if not results:
            return "No results found."
        page = wikipedia.page(results[0], auto_suggest=False)
        return page.summary[:1500]
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# The tool definition for Ollama
wikipedia_tool = {
    'type': 'function',
    'function': {
        'name': 'search_wikipedia',
        'description': 'Search Wikipedia for a given query and return a summary of the article.',
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'The search query to look up on Wikipedia.'
                }
            },
            'required': ['query']
        }
    }
}

def format_context(context_data):
    """Format the HotpotQA context into a readable string."""
    texts = []
    for title, sentences in zip(context_data['title'], context_data['sentences']):
        text = " ".join(sentences)
        texts.append(f"Title: {title}\nText: {text}")
    return "\n\n".join(texts)

def run_standard_generation(question, context, temp, prompt_type):
    if prompt_type == "standard":
        system_prompt = "Answer the question based strictly on the provided texts. Provide a concise answer."
    elif prompt_type == "cot":
        system_prompt = "Answer the question based strictly on the provided texts. Think step-by-step and write down your reasoning before providing the final answer."
    elif prompt_type == "detailed":
        system_prompt = "You are an expert factual Q&A system. Your task is to answer the following multi-hop question based strictly on the provided texts. Do not use outside knowledge. Identify the key entities, find the connecting information in the texts, and provide a clear, exact answer."
    
    user_prompt = f"Question: {question}\n\nTexts:\n{context}"
    
    response = ollama.chat(model='llama3.1', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ], options={'temperature': temp})
    
    return response['message']['content']

def run_agent_generation(question):
    messages = [
        {'role': 'system', 'content': 'You are a smart assistant. You must answer the following question. You have access to a Wikipedia search tool. Use it to find necessary information.'},
        {'role': 'user', 'content': f"Question: {question}"}
    ]
    
    # First call
    response = ollama.chat(
        model='llama3.1',
        messages=messages,
        tools=[wikipedia_tool],
        options={'temperature': 0.0}
    )
    messages.append(response['message'])
    
    # Tool execution loop (max 3 steps to prevent infinite loops)
    for _ in range(3):
        if not response['message'].get('tool_calls'):
            break
            
        for tool in response['message']['tool_calls']:
            if tool['function']['name'] == 'search_wikipedia':
                query = tool['function']['arguments'].get('query', '')
                tool_result = search_wikipedia(query)
                messages.append({
                    'role': 'tool',
                    'content': tool_result,
                    'name': 'search_wikipedia'
                })
                
        response = ollama.chat(
            model='llama3.1',
            messages=messages,
            tools=[wikipedia_tool],
            options={'temperature': 0.0}
        )
        messages.append(response['message'])
        
    return response['message']['content']

def main():
    print("Loading questions...")
    try:
        with open("data/sampled_100_questions.json", "r", encoding="utf-8") as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Dataset not found. Please run 01_prepare_dataset.py first.")
        return

    results = []
    if os.path.exists("data/raw_responses.json"):
        with open("data/raw_responses.json", "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                print(f"Wznowiono, znaleziono {len(results)} gotowych odpowiedzi.")
            except:
                pass
                
    processed_ids = {r["id"] for r in results}
    
    for i, q in enumerate(tqdm(questions, desc="Generating responses")):
        if q["id"] in processed_ids:
            continue
            
        question_text = q['question']
        context_text = format_context(q['context'])
        
        q_results = {
            'id': q['id'],
            'question': question_text,
            'true_answer': q['answer'],
            'type': q['type'],
            'level': q['level'],
            'responses': {}
        }
        
        # C1: Temp 0, Standard
        q_results['responses']['C1'] = [run_standard_generation(question_text, context_text, 0.0, "standard")]
        
        # C2: Temp 0.5, Standard (10x)
        q_results['responses']['C2'] = [run_standard_generation(question_text, context_text, 0.5, "standard") for _ in range(10)]
        
        # C3: Temp 1.0, Standard (10x)
        q_results['responses']['C3'] = [run_standard_generation(question_text, context_text, 1.0, "standard") for _ in range(10)]
        
        # C4: Temp 0, CoT
        q_results['responses']['C4'] = [run_standard_generation(question_text, context_text, 0.0, "cot")]
        
        # C5: Temp 0.5, CoT (10x)
        q_results['responses']['C5'] = [run_standard_generation(question_text, context_text, 0.5, "cot") for _ in range(10)]
        
        # C6: Temp 0, Detailed
        q_results['responses']['C6'] = [run_standard_generation(question_text, context_text, 0.0, "detailed")]
        
        # C7: Wikipedia Agent (Temp 0)
        q_results['responses']['C7'] = [run_agent_generation(question_text)]
        
        results.append(q_results)
        
        # Save incrementally
        if (i + 1) % 5 == 0:
            with open("data/raw_responses.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
    with open("data/raw_responses.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Generation complete. Saved to data/raw_responses.json")

if __name__ == "__main__":
    main()
