# Opis projektu: LLM Error Analysis (Analiza Błędów Modelu Językowego w pytaniach Multi-Hop)

> **Wysokopoziomowy opis projektu na zaliczenie (projekt grupowy dla 3 osób)**
> Poniższy dokument opisuje o co chodzi, jak działa system i jakie są oczekiwane cele końcowe. 
> 
> 💻 *Uwaga: Szczegółowy, techniczny opis działania skryptów krok po kroku oraz instrukcje uruchomienia znajdują się w osobnym pliku [README.md](README.md).*

---

## 📌 O co chodzi w projekcie
Projekt bada **jakie dokładnie błędy popełnia model językowy** (w tym wypadku otwartoźródłowy *Llama 3.1 8B*) odpowiadając na trudne, wieloskokowe pytania wymagające łączenia faktów. 

Nie pytamy tylko „ile razy model się pomylił”, lecz tworzymy zaawansowaną **mapę błędów**. Badamy:
- Czy model zmyśla (halucynuje)?
- Czy gubi się w rozumowaniu (multi-hop)?
- Czy myli podobne do siebie obiekty?
- Czy rodzaj błędów zmienia się, gdy zmieniamy sposób zadawania pytań (Prompt Engineering) lub gdy damy mu dostęp do zewnętrznych narzędzi (RAG / Agent)?

To klasyczne podejście z lingwistyki komputerowej zwane ***error analysis***. Zamiast jednej surowej liczby (np. „70% poprawnych”), dostarczamy szczegółowy profil błędów, ich częstość i to, od jakich parametrów zależą.

---

## 🔍 Na czym polegają pytania (Multi-hop QA)
Materiałem badawczym są pytania ze znanego akademickiego zbioru **HotpotQA**. Ich cechą charakterystyczną jest to, że odpowiedź wymaga **kilku kroków rozumowania** (połączenia informacji z dwóch różnych źródeł). 

*Przykładowo:* „W którym mieście urodził się reżyser filmu X?” 
Model musi najpierw zidentyfikować reżysera z jednego tekstu, a następnie znaleźć miejsce jego urodzenia w drugim tekście. Stąd nazwa *multi-hop* (wieloskokowe).

Do każdego pytania dołączony jest komplet tekstów encyklopedycznych: część z nich zawiera potrzebne fakty (złote akapity), a część to celowe rozpraszacze. Model dostaje pytanie razem z tymi tekstami i ma znaleźć odpowiedź — to test **rozumienia ze zrozumieniem**, a nie wiedzy wykutej na pamięć z internetu.

Pytania różnią się dwoma cechami, które śledzimy w wynikach:
- **Typ**: *bridge* (łańcuch faktów A→B→C) lub *comparison* (porównanie cech dwóch obiektów).
- **Trudność**: *easy*, *medium* lub *hard*.

---

## 🧊 Jak zbieramy dane (Zamrożony zestaw pytań)
Na samym początku system **losuje 100 pytań** w sposób zbalansowany z ogromnego zbioru treningowego, **zapisuje je do pliku** (`sampled_100_questions.json`) i dalej pracuje **wyłącznie na tym jednym, niezmiennym zbiorze**. 
Nigdy nie losujemy ponownie. Dzięki temu wszystkie późniejsze porównania (pomiędzy konfiguracjami lub modelami) dotyczą dokładnie tych samych pytań — co jest fundamentem rzetelności całego eksperymentu badawczego.

---

## ⚙️ Warianty zadawania pytań (7 Konfiguracji)
Te same 100 pytań zadajemy modelowi na **7 różnych sposobów**, aby zaobserwować, co wpływa na jego tok rozumowania i liczbę błędów. Zmieniamy trzy kluczowe osie:
1. **Temperaturę** — parametr „odwagi” modelu. Niska (0.0) to odpowiedzi przewidywalne i bezpieczne. Wysoka (1.0) to odpowiedzi bardziej kreatywne, ale podatne na halucynacje.
2. **Styl polecenia (Prompt)** — od suchego „odpowiedz krótko”, przez zaawansowaną prośbę o **rozumowanie krok po kroku** (*Chain-of-Thought*), po wersję z rozbudowaną instrukcją systemową dla eksperta.
3. **Tryb pracy (Agent / RAG)** — w większości przypadków model dostaje potrzebne teksty od razu na tacy. Wariatem eksperymentalnym jest tryb, gdzie model sam musi **wyszukać** informacje w Wikipedii używając wbudowanego narzędzia (Function Calling).

Dla konfiguracji niedeterministycznych (temperatura > 0) zadajemy każde pytanie **10 razy**. Otrzymujemy w ten sposób **rozkład odpowiedzi**, co pozwala nam podzielić błędy na:
- **Błąd systematyczny** — model myli się za każdym razem w ten sam sposób (silne, błędne przekonanie).
- **Błąd stochastyczny** — model czasem trafia, czasem nie (jest niestabilny). Błędy stochastyczne dają się łatwo wyleczyć techniką *Self-Consistency*.

---

## 📊 Klasyfikacja i Taksonomia Błędów

Każda odpowiedź jest najpierw oceniana przez 3 niezależne metryki: Exact Match (surowe dopasowanie), F1 Score (pokrycie słów) oraz inteligentny **LLM-as-a-judge** (np. Gemini 1.5 Pro). Jeśli sędzia uzna, że odpowiedź jest zła, klasyfikuje błąd do 1 z 5 kategorii:

1. **Halucynacja** — odpowiedź nie wynika z dostarczonych tekstów.
2. **Błąd rozumowania wieloskokowego** — model odnalazł dobrze pierwszy fakt, ale wysnuł zły wniosek w drugim kroku.
3. **Błąd formatu** — model znał odpowiedź, ale podał ją w złej postaci.
4. **Błąd negacji** — przeoczenie słowa „nie” w kontekście.
5. **Pomylenie encji** — wybór powiązanego, ale błędnego obiektu z tekstu.

### 6. 👩‍🔬 Ręczna Walidacja (Złoty Standard dla Zespołu)
Aby potwierdzić przed wykładowcą, że nasz algorytm sędziowski jest poprawny, przeprowadzamy walidację jakości na małej próbce. Wyznaczona jedna osoba z zespołu dokonuje ślepej adnotacji losowej próby 100 błędów w pliku CSV (`adnotacja_reczna.csv`).

### 7. 🧮 Wyliczenie Cohen's Kappa
Po uzupełnieniu adnotacji uruchamiany jest dedykowany skrypt (`07_calculate_kappa.py`), który zestawia odpowiedzi człowieka z AI i liczy statystyczny współczynnik zgodności w ocenie błędów językowych LLMa.

### 8. 📊 Stworzenie Prezentacji Zaliczeniowej
Ostatnim krokiem jest zebranie 3 wygenerowanych wykresów z folderu `wyniki_i_wnioski`, wyniku Kappa z Etapu 7 oraz merytorycznych konkluzji (plik `RAPORT_KONCOWY.md`) na zgrabne slajdy. Główny akcent prezentacji musi paść na dowiedzenie, jak bardzo przestarzałe są klasyczne metryki (EM/F1) oraz z czego wynika dominacja błędów "Multi-hop" w Llama 3.1 8B.

---

## 🎯 Cele Projektu i Co trafi na Prezentację
Zwieńczeniem całego potoku analitycznego jest wygenerowanie twardych danych i pięknych wykresów, które trafią na końcową prezentację zaliczeniową. Oto główne *deliverables*:

1. **Profil Błędów (Error Profile)**: Wykresy kołowe (Pie Charts) z modułu analizy, pokazujące jednoznacznie, na czym potyka się badany model (np. okaże się, że 60% błędów to 'Pomylenie encji', a tylko 10% to klasyczna 'Halucynacja').
2. **Skuteczność Prompt Engineeringu**: Wykres słupkowy obrazujący wpływ temperatury oraz trybu *Chain-of-Thought* na redukcję błędów.
3. **Analiza Agentowa**: Wniosek płynący z testu wyszukiwarki (Wikipedia Agent). Na ile samodzielne szukanie pogarsza jakość względem podanych na tacy dokumentów.
4. **Błędy Systematyczne vs Stochastyczne**: Wykres typu stacked-bar udowadniający wprost, że część błędów to tylko losowe potknięcia modelu, a reszta to poważne wady wynikające z architektury.
5. **Zgodność Human vs AI**: Konkretna liczba udowadniająca na slajdzie metodykę oceny modeli za pomocą innych, potężniejszych modeli (LLM-as-a-judge).
