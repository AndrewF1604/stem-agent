# Status Projektu: LLM Error Analysis

Ten plik ułatwi współpracę w zespole lub z kolejnymi agentami. Pokazuje on dokładnie, co zostało już zrobione, a co jest dopiero przed Wami.

## 🟢 Etap 1: Przygotowanie Danych (ZAKOŃCZONO)
- **Skrypt:** `01_prepare_dataset.py`
- **Status:** Wygenerowano plik `data/sampled_100_questions.json`.
- Mamy wylosowaną, zbalansowaną i "zamrożoną" próbkę 100 pytań HotpotQA (easy, medium, hard).

## 🟢 Etap 2: Generowanie Odpowiedzi (ZAKOŃCZONO)
- **Skrypt:** `02_generate_responses.py`
- **Status:** Wygenerowano wszystkie 100 pytań! Zapisano potężny plik `data/raw_responses.json`. M5 przetrwał wielogodzinny maraton z Llama 3.1. Praca lokalnego procesora na tak dużą skalę została oficjalnie i z pełnym sukcesem zakończona.

## 🟡 Etap 3: Ewaluacja Odpowiedzi (GOTOWE DO ODPALENIA PRZEZ ZESPÓŁ)
- **Skrypt:** `03_evaluate_metrics.py`
- **Status logiki:** Zaktualizowano API z przestarzałego `gemini-1.5` na nowiutkie, oficjalnie wspierane `gemini-2.5-flash`. Dodano też algorytm "Exponential Backoff", dzięki czemu w razie darmowych limitów (rate-limit) skrypt bezpiecznie zaczeka i ponowi próbę bez wywalania błędu (brak błędu 404/429).
## 🟢 Etap 3: Automatyczna Ewaluacja (GOTOWE - Zakończono używając lokalnego sędziego Gemma 2)
Skrypt: 03_evaluate_metrics.py
Oblicza EM, F1 oraz ocenę "LLM as a judge" (czy odpowiedź jest merytorycznie poprawna pomimo formy).

## 🟢 Etap 4: Klasyfikacja Błędów (GOTOWE - Użyto Gemma 2 do kategoryzacji lingwistycznej)
Skrypt: 04_classify_errors.py
Dla odpowiedzi z oceną "LLM-judge = 0", skrypt kategoryzuje błąd (halucynacja, błąd formatu, błąd rozumowania, itp.) oraz analizuje stochastyczność (czy błąd jest powtarzalny).

## 🟢 Etap 5: Analiza i Wykresy (GOTOWE - Zapisane w folderze data/)
Skrypt: 05_analyze_results.py
Generuje pliki graficzne (wykresy słupkowe, macierze) z wynikami do wklejenia w raporcie.

## 🟡 Etap 6: Walidacja Ludzka (W TRAKCIE - Czeka na człowieka)
- **Skrypt:** Brak (Ręczna robota w Excelu)
- Otwórz plik `data/manual_annotation/INSTRUKCJA_ADNOTACJI.md` – tam jest dokładna rozpiska, jak przypisywać cyferki 1-5.
- Pamiętaj, aby uzupełnić kolumnę "Twoja Klasyfikacja (wpisz 1-5)" w pliku `adnotacja_reczna.csv`! Zapisz plik.

## 🔴 Etap 7: Wyliczenie Cohen's Kappa (CZEKA NA ZAKOŃCZENIE ETAPU 6)
- **Skrypt:** `07_calculate_kappa.py`
- Gdy wypełnicie już ręcznie próbkę z Etapu 6, po prostu odpalcie ten skrypt w terminalu: `python 07_calculate_kappa.py`. 
- Automatycznie połączy on Wasze oceny z ukrytymi ocenami Gemmy i wypluje gotowy, naukowy wskaźnik zgodności do raportu!

## 🔴 Etap 8: Przygotowanie Prezentacji Zaliczeniowej (CZEKA)
Otwórzcie plik `wyniki_i_wnioski/RAPORT_KONCOWY.md`. Na jego podstawie złóżcie ok. 6-8 slajdów:
1. **Wstęp i metodologia** (zbiór HotpotQA, odpytywanie 7 konfiguracji Llama 3.1 w Ollamie).
2. **Kryzys klasycznych metryk** (Wykres 1 - pokazanie, dlaczego Exact Match = 0% a "LLM-as-a-judge" działa).
3. **Typologia błędów** (Wykres 2 - udowodnienie przewagi błędów Multi-hop reasoning nad np. zwykłymi halucynacjami).
4. **Błędy stochastyczne a systematyczne** (Wykres 3 - zjawisko powtarzalności błędu w modelach LLM).
5. **Wynik z Walidacji Ludzkiej (Etap 7)** (Zrzut ekranu wyniku konsoli pokazującego % dokładności i wskaźnik Cohen's Kappa - dowód na naukową wiarygodność sędziego).
6. **Wnioski z Case Study** (np. że nawet mocarne modele dają się czasem nabrać na "lanie wody").
