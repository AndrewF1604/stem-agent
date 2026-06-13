# Projekt: LLM Error Analysis

Ten projekt służy do ewaluacji otwartoźródłowych modeli językowych (np. Llama 3.1 8B) w zadaniach opartych na złożonym rozumowaniu (Multi-Hop QA). Skupia się na architekturze i dystrybucji błędów, analizując wpływ temperatury, prompt engineeringu oraz RAG-a na typy popełnianych przez model pomyłek.

## Architektura i Przepływ Danych (Dla kolejnych agentów)

Projekt podzielony jest na 5 modularnych skryptów Pythona, które uruchamia się sekwencyjnie. Dane wejściowe przechodzą przez kolejne pliki JSON w katalogu `data/`.

> [!IMPORTANT]  
> Przed uruchomieniem czegokolwiek, upewnij się, że masz aktywne środowisko `venv`, zainstalowane biblioteki z `requirements.txt` oraz serwer Ollama z pobranym modelem (`ollama pull llama3.1`).

### 1. `01_prepare_dataset.py`
Pobiera oryginalny zbiór `hotpot_qa` (wariant `distractor`, split `train`) z biblioteki Hugging Face `datasets`.
- Losuje dokładnie 100 pytań (po 25 z każdej kombinacji typu: `bridge`/`comparison` oraz trudności: `easy`/`hard`).
- Zapisuje zamrożony, niemutowalny zbiór testowy do pliku **`data/sampled_100_questions.json`**.

### 2. `02_generate_responses.py`
Serce systemu. Przechodzi przez wszystkie 100 pytań i pyta model `llama3.1` na 7 sposobów:
- C1: Temp 0.0, Standard Prompt
- C2: Temp 0.5, Standard Prompt (10 iteracji)
- C3: Temp 1.0, Standard Prompt (10 iteracji)
- C4: Temp 0.0, Chain-of-Thought (CoT) Prompt
- C5: Temp 0.5, Chain-of-Thought (CoT) Prompt (10 iteracji)
- C6: Temp 0.0, Detailed Expert Prompt
- C7: Agent z dostępem do narzędzia przeszukiwania Wikipedii (Python `wikipedia` API via Tool Calling, max 3 iteracje).
Odpowiedzi są dopisywane do **`data/raw_responses.json`**.

### 3. `03_evaluate_metrics.py`
Liczy metryki poprawności dla wygenerowanych odpowiedzi:
1. **Exact Match (EM)** - sztywne, dokładne dopasowanie stringów po usunięciu interpunkcji i sprowadzeniu do małych liter.
2. **F1 Score** - częściowe pokrycie tokenów (słów).
3. **LLM-as-a-judge** - Sędzia-model decydujący, czy odpowiedź jest faktualnie poprawna, mimo zmian formatu czy parafraz.
Zapisuje poszerzone dane do **`data/evaluated_responses.json`**.

### 4. `04_classify_errors.py`
Przechodzi przez te odpowiedzi, które sędzia ocenił na "0" (błędne) i prosi model o przydzielenie ich do 1 z 5 klas błędu:
- Halucynacja
- Błąd rozumowania wieloskokowego
- Błąd formatu
- Błąd negacji
- Pomylenie encji
Dokonuje również analizy tego, czy błąd był *systematyczny* (we wszystkich iteracjach ten sam błąd) czy *stochastyczny*.
Wynik trafia do **`data/classified_errors.json`**.

### 5. `05_analyze_results.py`
Skrypt używający `pandas` i `matplotlib` do agregacji danych z klasyfikacji. Generuje wykresy słupkowe i kołowe (trafiają do katalogu `data/`), które podsumowują: 
- Średnią jakość odpowiedzi na konfigurację.
- Dystrybucję błędów.
- Stosunek błędów systematycznych do stochastycznych.

### 6. `06_prepare_manual_annotation.py` (Walidacja na ludziach)
- Bierze wszystkie stwierdzone błędy i losuje z nich reprezentatywną próbkę 100 sztuk.
- Formatuje je do pliku CSV (`data/manual_annotation/adnotacja_reczna.csv`), ukrywając klasyfikację AI.
- Służy do ślepej adnotacji ręcznej przez wyznaczoną osobę w celu wyliczenia miary zgodności i udowodnienia rzetelności sędziego AI.

### 7. `07_calculate_kappa.py` (Zgodność człowieka z AI)
- Oczekuje wypełnionego przez badacza pliku `adnotacja_reczna.csv`.
- Zestawia ręczne oceny (1-5) z ukrytym kluczem oceny wygenerowanym przez Gemmę.
- Wylicza procentową dokładność (Accuracy) oraz współczynnik Cohen's Kappa, określający poziom wiarygodności sędziego.

### 8. Utworzenie Prezentacji Zaliczeniowej (Finał)
- Zespół wykorzystuje wykresy z folderu `wyniki_i_wnioski/` oraz wyliczoną w Etapie 7 statystykę Cohen's Kappa do zbudowania końcowych slajdów na obronę projektu.
- Plik `wyniki_i_wnioski/RAPORT_KONCOWY.md` stanowi bazę merytoryczną do opowiedzenia o zjawiskach takich jak Multi-hop reasoning errors, zapaść metryk n-gramowych czy halucynacje.

## Cele Projektu i Oczekiwane Wyniki
- Zrozumienie na jakich pytaniach 8-miliardowy model się poddaje.
- Sprawdzenie skuteczności "Chain-of-Thought" oraz RAG-a na otwartej architekturze błędu.
- Zbudowanie potężnej prezentacji na koniec projektu opierającej się o wykresy wygenerowane przez `05_analyze_results.py`.

## UWAGI DOTYCZĄCE MODELU SĘDZIEGO (LLM-as-a-Judge)

W wersji początkowej skrypty `03_evaluate_metrics.py` i `04_classify_errors.py` odwołują się do wywołań w Ollamie dla modelu `llama3.1`. Dla uzyskania rzetelnych metryk naukowych, faza oceny i klasyfikacji (skrypty 3 i 4) powinna być obsługiwana przez silniejszy model (np. Gemini 1.5 Pro lub GPT-4o), aby model 8B nie "oceniał samego siebie". Wszelkie zmiany sędziego należy wprowadzić podmieniając wywołania `ollama.chat` w tych dwóch plikach na odpowiednie API mocniejszego modelu.
