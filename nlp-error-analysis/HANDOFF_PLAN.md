# Plan dalszych prac — co zostało do zrobienia

> Setup i opis tego co już istnieje: `HANDOFF_SETUP.md`. Ten plik to roadmap od stanu obecnego do oddania.

## Pytania badawcze

1. **PB1**: Jak grupują się błędy LLM w taksonomii lingwistyczno-semantycznej? Jaka jest ich dystrybucja na 100 pytań?
2. **PB2**: Czy dystrybucja błędów koreluje z typem pytania (bridge vs comparison) i trudnością (easy vs hard)?
3. **PB3**: Czy poszczególne conditions (temperatura, CoT, verbose, search) zmieniają dystrybucję błędów? W jaki konkretny sposób?
4. **PB4**: Jak zgodne są trzy metryki (EM, F1, LLM-Judge)? Gdzie się rozjeżdżają i co to mówi o jakości każdej?
5. **PB5** *(nowe, z designu iteracyjnego)*: Czy błędy Llamy są **systematyczne czy stochastyczne**? Ile z nich "naprawia" self-consistency (głosowanie większościowe z K próbek)?

## Roadmap (Fazy 1–3 ✅, redesign iteracyjny ✅, dalej 4–7)

### Faza 4 — Pełny sweep + ewaluacja (~3-4h pracy, ~3h wallclock)

> **Zasada zbierania danych:** na samym początku **losujemy raz 100 pytań**, **zapisujemy je do pliku** i całą resztę badania prowadzimy **wyłącznie na tym jednym, niezmiennym zbiorze**. Nigdy nie losujemy ponownie — wszystkie conditions i wszystkie iteracje pytają o dokładnie te same `qid`. To warunek rzetelności (paired design) i jest pilnowany guardem na hash zestawu.

**4.1 Wygeneruj zamrożony zestaw (jeśli nie ma)**
```powershell
cd c:\Users\Andrzej\Desktop\aws\nlp-error-analysis
.\.venv\Scripts\python.exe dataset.py    # losuje 100 pytań -> zapisuje data/samples_100.json + manifest z hashem
```
Od tego momentu plik `data/samples_100.json` jest źródłem prawdy — `sweep.py` czyta go i tylko jego. Jeśli kiedykolwiek go przegenerujesz innym seedem, guard na hash przerwie wznawiany sweep, żeby nie zmieszać dwóch zestawów.

**4.2 Pełny sweep**
```powershell
.\.venv\Scripts\python.exe sweep.py       # ~3700 inferencji, resume działa
```
- **Nie usypiać laptopa** w trakcie
- Guard na hash chroni przed pomieszaniem zestawów

**4.3 Ewaluacja**
```powershell
.\.venv\Scripts\python.exe evaluate_all.py
```
- Output: `analysis/raw_long.csv` (per próbka) + `analysis/per_question.csv` (agregaty)
- Judge dedupowany → dużo mniej wywołań niż 3700

**4.4 Sanity check**
- `n_error` w per_question.csv niski (<5%)? jeśli nie — debug
- accuracy per condition się różni (CoT vs temp=1.0)?
- G_agent_search ma sensowne odpowiedzi (DDG)?
- `agreement` przy temp=0 ≈ 1.0 (sanity: determinizm); przy temp=1.0 wyraźnie niższy

### Faza 5 — Taksonomia + anotacja błędów (~6-8h)

**5.1 Stwórz `taxonomy.py`** — 5 kategorii błędów:
1. **Halucynacja** — odpowiedź nie wynika z kontekstu/pamięci
2. **Multi-hop reasoning failure** — pierwszy fakt OK, zły wniosek w drugim kroku
3. **Parsing/format error** — znał odpowiedź, zły format
4. **Negation error** — przeoczył negację
5. **Entity confusion** — semantycznie powiązana ale zła encja

Każda: nazwa, opis 2-3 zdania, 3-5 przykładów *jak rozpoznać* (idą do appendiksu).

**Oś dodatkowa z PB5**: każdy błąd taguj też jako **systematyczny** (`wrong_consistency` wysoka — model uparcie ta sama zła odp) vs **stochastyczny** (niska — losowe zgadywanie). To rozcina taksonomię wzdłuż nowego wymiaru.

**5.2 Filtruj błędy** — jednostką jest teraz *pytanie* (`per_question.csv`), nie pojedyncza odpowiedź:
```python
import pandas as pd
pq = pd.read_csv("analysis/per_question.csv")
# pytania gdzie majority vote jest zły:
errors = pq[pq["majority_judge"] == 0]
# albo twarde błędy (model nigdy nie trafił w K prób):
hard_errors = pq[pq["judge_passany"] == 0]
errors.to_csv("analysis/errors_to_annotate.csv", index=False)
```
Do treści odpowiedzi/trajektorii sięgaj po `raw_long.csv` (per próbka).

**5.3 Manualna anotacja ~50 błędów** — stratified po condition (~7/condition). CLI lub Streamlit pokazujący: pytanie, gold, odpowiedź(i) modelu, kontekst/trajektorię, judge reasoning, **rozkład K próbek** (agreement/entropy). Anotator wybiera kategorię + flagę systematyczny/stochastyczny → `analysis/manual_annotations.csv`.

**5.4 Auto-anotacja** — `taxonomy.py` ma `categorize_auto(question, gold, answer, judge_reason) -> category` (prompt z opisami kategorii + few-shot z 5.1). Uruchom na błędach → `analysis/auto_annotations.csv`.

**5.5 Cohen's kappa manual ↔ auto**
```python
from sklearn.metrics import cohen_kappa_score
merged = manual.merge(auto, on="qid", suffixes=("_m","_a"))
kappa = cohen_kappa_score(merged["category_m"], merged["category_a"])
```
Cel: kappa ≥ 0.6. Niżej → sprecyzuj taksonomię.

### Faza 6 — Analiza + wykresy (~5-6h)

`analysis.py` + `plots.py`. **`plots.py` generuje każdy wykres w dwóch wersjach**: `report/` (gęsta, z CI, do PDF) i `slides/` (rzadka, duża czcionka, podświetlony 1 element) — parametr `style=`. Stała paleta: 1 kolor per condition, 1 per kategoria błędu, trzymana wszędzie.

**Klasyczne (per condition):**
- 6.1 Accuracy + 95% CI (Wilson) dla EM/F1/Judge per condition — tabela headline 7×3
- 6.2 **Hero figure**: stacked bar dystrybucji 5 kategorii błędów per condition (PB1)
- 6.3 Heatmapa condition × question_type (PB2)
- 6.4 Heatmapa condition × difficulty (PB2)
- 6.5 Heatmapa error_category × question_type (PB2)
- 6.6 Inter-metric agreement: EM vs F1 vs Judge (PB4) — UpSet/Venn + liczby false-neg EM
- 6.7 Wpływ temperatury (A→B→C)
- 6.8 Wpływ CoT (A↔D, B↔E) — nie tylko accuracy, ale *zmiana profilu błędów*
- 6.9 Agent vs context (A↔G)

**Nowe z designu iteracyjnego (PB5):**
- 6.10 **Self-consistency lift**: `majority_judge` accuracy − `judge_pass1` per condition (Wang et al. 2022). Czy głosowanie podnosi trafność?
- 6.11 **Kalibracja przez zgodę** (bez logitów): reliability diagram — bin po `agreement`, oś Y empiryczna trafność `majority`. Policz ECE. Empiryczna wersja "calibration in LLMs" (Tian et al. 2023).
- 6.12 **pass@1 vs pass@any gap**: capability vs stability. Duża luka = umie ale niestabilny; mała luka przy niskim pass@any = nie wie.
- 6.13 **Systematyczne vs stochastyczne błędy**: histogram `wrong_consistency` dla pytań `judge_passany==0`. + 2 przykłady jakościowe (uparcie zła odp vs rozsypka).
- 6.14 **Temperatura × entropia**: `answer_entropy` vs temp (A→B→C). Oczekiwane: entropia rośnie, ale `majority_judge` zostaje płaskie → self-consistency odzyskuje trafność.

**Testy parowane** (z paired design): McNemar dla par conditions (A↔D, A↔G…), Cochran's Q dla wszystkich 7, paired bootstrap CI na różnicy accuracy.

### Faza 7 — Prezentacja przed grupą + raport

#### 7a. Prezentacja przed grupą (~12-14 slajdów, ~12 min)

Jeden slajd = jedna myśl = jeden wizual. Zero tabel 7×3 na ekranie. Każdy slajd recykluje figurę z `plots/slides/`.

```
 1. Tytuł — hook ("Jakie błędy popełnia Llama 3.1 8B w multi-hop QA")
 2. Problem w 1 obrazku — 1 pytanie HotpotQA, dwuskokowa natura
 3. Setup w 1 grafice — 100 pytań → 7 conditions → K iteracji → 3700 odp → 3 metryki
 4. 7 conditions — ikonografia (temp/CoT/agent)
 5. Headline — accuracy per condition (bar)
 6. HERO: dystrybucja błędów (PB1) — najwięcej czasu mówienia
 7. Przykład błędu na żywo — multi-hop failure (lub demo drill-down)
 8. Wpływ temperatury (PB3)
 9. Wpływ CoT (PB3) — "CoT zmienia PROFIL błędów, nie tylko accuracy"
10. Agent vs context (PB3)
11. Systematyczny vs stochastyczny (PB5) — teza całości: "różne ustawienia zmieniają nie ILE, a JAK model się myli"
12. Metryki się nie zgadzają (PB4) + kappa
13. 3 findingi + limitacje (szczerość = wiarygodność)
14. Dziękuję / pytania
+ slajdy BACKUP za końcem (pełne heatmapy, tabela CI, reliability diagram, kappa) na pytania z sali
```
Opcjonalny mocny akcent: **demo na żywo** drill-down w jeden błąd (trajektoria agenta + reasoning Judge'a) ze Streamlita z 5.3 — 30s demo > 3 slajdy, ale tylko jeśli stabilne.

#### 7b. Raport (~3000 słów, polski, + bibliografia + appendix)

```
1. Wstęp (300-400) — multi-hop QA, tradycja error analysis, PB1-PB5, kontrybucja
2. Praca powiązana (400-500) — HotpotQA (Yang 2018), ReAct (Yao 2022), CoT (Wei 2022),
   Self-Ask (Press 2023), LLM-as-Judge (Zheng 2023), self-consistency (Wang 2022),
   pass@k (Chen 2021), calibration (Tian 2023), contamination (Magar & Schwartz 2022)
3. Metodologia (700-900)
   3.1 Dataset + stratified sampling (100 = 25×4, zamrożony zestaw)
   3.2 Model: Llama 3.1 8B przez Ollama
   3.3 Siedem conditions (tabela z K)
   3.4 Design repeated-measures: K iteracji, paired qid, seed per iter
   3.5 Trzy metryki: EM, F1, LLM-Judge (+ normalizacja, walidacja Judge przez kappa)
   3.6 Metryki rozkładu: agreement, entropia, pass@1/@k, self-consistency, wrong_consistency
   3.7 Taksonomia błędów (5 kategorii + oś systematyczny/stochastyczny)
4. Wyniki (900-1100 + 6-8 wykresów)
   4.1 Headline accuracy per condition
   4.2 Dystrybucja błędów (hero)
   4.3 Temperatura, 4.4 CoT, 4.5 agent vs context
   4.6 Cross-taby condition × {q_type, difficulty}
   4.7 Self-consistency lift + pass@1 vs pass@any (PB5)
   4.8 Kalibracja przez zgodę + temperatura × entropia (PB5)
   4.9 Inter-metric agreement (PB4) + kappa manual↔auto
5. Dyskusja (500-700)
   5.1 Co dystrybucje mówią o multi-hop reasoning w Llama
   5.2 Dlaczego CoT (nie) pomaga
   5.3 Agent vs context
   5.4 Systematyczne vs stochastyczne błędy — co naprawia self-consistency
   5.5 Inter-metric disagreement a kalibracja
   5.6 Ograniczenia: 25/cell (umiarkowane), pojedynczy anotator, Llama jako model+sędzia,
       contamination, DDG variability w G (część wariancji wewnątrz-pytaniowej to nie sampling)
6. Wnioski (200-300) — top 3 findings, implikacje, future work
7. Bibliografia (~12-15)
Appendix A: 5 kategorii błędów z przykładami
Appendix B: Annotation guidelines
```

## Harmonogram (8 dni)

| Dzień | Zadanie | h |
|---|---|:-:|
| 1 | dataset.py (zamrożony zestaw), sweep (~3h w tle), evaluate_all | 4 |
| 2 | Sanity check, draft taksonomii (+ oś systematyczny/stochastyczny) | 4 |
| 3 | Manualna anotacja ~50 błędów | 5 |
| 4 | Auto-anotacja + kappa | 4 |
| 5 | Statystyki + wykresy (report + slides) | 6 |
| 6 | Slajdy prezentacji + raport sekcje 1-3 | 5 |
| 7 | Raport sekcje 4-5 | 5 |
| 8 | Sekcje 6-7, próba prezentacji, finalizacja | 4 |

## Realne ryzyka i mitigations

| Ryzyko | Mitigation |
|---|---|
| Sweep się wywala (Ollama/DDG) | Resume po udanych próbkach; pull innego modelu |
| Pomieszanie zestawów pytań | Guard na hash przerywa bieg; manifest jako źródło prawdy |
| Kappa < 0.5 | Sprecyzuj taksonomię, więcej few-shot w auto-annotatorze |
| Słabe wyniki overall (<50%) | qwen2.5:14b (9GB) jeśli starczy VRAM |
| Sweep > 3h | Resume; zmniejsz K dla stochastycznych (`--iters 5`) |
| Awaria demo na żywo | Fallback: statyczny przykład na slajdzie 7 |
| Brak czasu | Skróć Discussion + Bibliografię, zostaw Wyniki obszerne |

## Pliki które jeszcze NIE istnieją

- `taxonomy.py` — definicje kategorii + auto-categorizer + oś systematyczny/stochastyczny
- `annotate.py` — CLI/Streamlit do manualnej anotacji (+ tryb "explore" pod demo)
- `analysis.py` — statystyki, cross-taby, testy parowane (McNemar/Cochran)
- `plots.py` — wykresy w dwóch stylach (report/slides)
- `raport.md`/`raport.tex` + `slides/` (deck)

## Dla nowej konwersacji Claude

Daj te dwa pliki jako pierwszy input:
1. `HANDOFF_SETUP.md` — co jest, jak uruchomić
2. `HANDOFF_PLAN.md` — co dalej

Plus: "Pracuję nad projektem na kurs NLP. Stan i plan w dwóch plikach. Aktualnie [gdzie jesteś]."
