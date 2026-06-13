# Instrukcja Ręcznej Adnotacji (Walidacja Sędziego AI)

W tym folderze znajduje się plik `adnotacja_reczna.csv`. Zawiera on 100 wylosowanych, błędnych odpowiedzi wygenerowanych przez model Llama 3.1. 
Twoim zadaniem jest ocenić, **jakiego rodzaju błąd** popełnił model w każdym z przypadków.

## Jak to zrobić krok po kroku:
1. Otwórz plik `adnotacja_reczna.csv` w programie Excel, Numbers lub Google Sheets.
2. Przeczytaj uważnie `Pytanie`, `Złotą Odpowiedź` (czyli to, co jest faktycznie prawdą według bazy) oraz `Odpowiedź Modelu (Błędna)`.
3. Jeśli potrzebujesz sprawdzić kontekst źródłowy, masz do dyspozycji kolumnę ze `Skróconym Kontekstem`.
4. Wpisz odpowiednią **cyfrę od 1 do 5** w ostatniej, pustej kolumnie o nazwie `Twoja Klasyfikacja (wpisz 1-5)`.

---

## Klucz do klasyfikacji (Co oznaczają cyfry):

**1 - Halucynacja (Hallucination)** 
*Model podał informację całkowicie zmyśloną, której w ogóle nie ma w podanym tekście, lub zaczął generować urojone komendy systemowe.*

**2 - Błąd rozumowania wieloskokowego (Multi-hop reasoning error)** 
*Zdecydowana większość trudnych pytań. Pytanie wymagało połączenia faktów z dwóch tekstów. Model poprawnie zidentyfikował pierwszy krok (np. znalazł nazwisko reżysera w tekście A), ale zawiódł na drugim kroku (np. nie potrafił znaleźć w tekście B, gdzie reżyser się urodził).*

**3 - Błąd formatu (Format error)** 
*Model znał prawidłowe fakty, ale zaprezentował je w kompletnie złej formie. Np. zaczął tłumaczyć swój proces myślowy krok po kroku na dwa akapity (lanie wody), zamiast po prostu odpowiedzieć jednym, konkretnym słowem, czego wymagał Ground Truth.*

**4 - Błąd negacji (Negation error)** 
*Model przeoczył lub odwrotnie zinterpretował słowo oznaczające negację w tekście źródłowym (rzadki błąd).*

**5 - Pomylenie encji (Entity confusion)** 
*Model znalazł w tekście osobę, miasto lub datę z powiązanego kontekstu, ale przypisał je błędnie. Np. w pytaniu o aktora grającego rolę główną, model udzielił poprawnej odpowiedzi dotyczącej aktora drugoplanowego z tego samego filmu.*

---

## Co dalej?
Gdy uzupełnisz wszystkie 100 wierszy i zapiszesz plik CSV (koniecznie w formacie `.csv`!):
Wróć do głównego folderu z kodem i w terminalu uruchom skrypt:
`python 07_calculate_kappa.py`

Skrypt w ułamku sekundy porówna Twoje oceny ludzkie z tym, jak ocenił to niezależnie nasz sędzia AI (Gemma 2 9B). Wypluje gotową statystykę Cohen's Kappa na ekran, którą po prostu przekleicie na slajd końcowy prezentacji. Powodzenia!
