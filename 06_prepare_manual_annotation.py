import json
import pandas as pd
import random
import os

def main():
    try:
        with open("data/classified_errors.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("data/classified_errors.json not found. You must run steps 1-4 first.")
        return
        
    try:
        with open("data/sampled_100_questions.json", "r", encoding="utf-8") as f:
            questions = {q['id']: q for q in json.load(f)}
    except FileNotFoundError:
        print("data/sampled_100_questions.json not found.")
        return

    # Zbieramy wszystkie BŁĘDNE odpowiedzi z danych
    wrong_answers = []
    
    for item in data:
        q_id = item['id']
        question_text = item['question']
        truth = item['true_answer']
        
        # Pobieramy oryginalny kontekst (tylko teksty, bez szumu)
        context_data = questions[q_id]['context']
        context_texts = [f"{t}: {' '.join(s)}" for t, s in zip(context_data['title'], context_data['sentences'])]
        context_preview = "\n".join(context_texts)[:1500] + "... (skrócone)"
        
        for config, metrics_list in item['metrics'].items():
            for i, metric in enumerate(metrics_list):
                if metric['judge'] == 0:
                    model_prediction = item['responses'][config][i]
                    gemini_classification = item['errors'][config]['classifications'][i]
                    
                    wrong_answers.append({
                        'ID Pytania': q_id,
                        'Konfiguracja': config,
                        'Pytanie': question_text,
                        'Złota Odpowiedź': truth,
                        'Odpowiedź Modelu (Błędna)': model_prediction,
                        'Skrócony Kontekst': context_preview,
                        'Twoja Klasyfikacja (wpisz 1-5)': '',  # Puste, do wypełnienia przez człowieka
                        'Klasyfikacja Gemini (UKRYTE przed adnotatorem)': gemini_classification
                    })

    # Tasujemy błędne odpowiedzi, żeby losowo je rozdzielić
    random.seed(42)
    random.shuffle(wrong_answers)
    
    # Wybieramy próbkę 100 błędów (lub mniej, jeśli model popełnił mniej błędów)
    sample_size = min(100, len(wrong_answers))
    sampled_errors = wrong_answers[:sample_size]
    
    # Zapisujemy do CSV dla jednej osoby
    os.makedirs("data/manual_annotation", exist_ok=True)
    
    df = pd.DataFrame(sampled_errors)
    
    # Tworzymy wersję ślepą dla człowieka (usuwamy kolumnę Gemini)
    blind_df = df.drop(columns=['Klasyfikacja Gemini (UKRYTE przed adnotatorem)'])
    blind_df.to_csv(f"data/manual_annotation/adnotacja_reczna.csv", index=False, encoding="utf-8-sig")
    
    # Tworzymy wersję z kluczem odpowiedzi (dla nas na później)
    df.to_csv(f"data/manual_annotation/KLUCZ_adnotacja_reczna.csv", index=False, encoding="utf-8-sig")
        
    print(f"Wygenerowano plik 'adnotacja_reczna.csv' (zawierający próbkę {sample_size} błędów) w folderze 'data/manual_annotation'.")
    print("Osoba odpowiedzialna za adnotację wypełnia plik cyframi 1-5. Plik KLUCZ służy tylko do weryfikacji po zakończeniu adnotacji!")

if __name__ == "__main__":
    main()
