# Setup i obecny stan projektu — Analiza błędów LLM na HotpotQA

> Ten plik opisuje co już istnieje i jak to uruchomić. Plan dalszych prac jest w `HANDOFF_PLAN.md`, a wysokopoziomowy opis projektu bez szczegółów technicznych w `OPIS_PROJEKTU.md`.
> Kod projektu mieszka w `c:\Users\Andrzej\Desktop\aws\nlp-error-analysis\` — te markdowny to wersjonowana kopia dokumentacji.

## Cel projektu

Projekt na kurs NLP (5. rok teleinformatyki). Analizuję jakie błędy popełnia Llama 3.1 8B odpowiadając na pytania multi-hop z HotpotQA pod różnymi konfiguracjami (temperatura, prompt, dostęp do wyszukiwarki). Raport po polsku, ~3000 słów, w stylu klasycznego NLP error analysis paper. Forma zaliczenia: **prezentacja projektu przed grupą** + raport.

Konsultacja z prowadzącym ustaliła:
- **Ewaluacja kilkuset odpowiedzi**
- **Różne zmiany**: temperatura, prompt, wyszukiwarka
- **Wykazać dystrybucje** błędów per condition
- **Eksplicytnie opisać** metodologię oceny "która odp jest dobra/zła"

## Design: repeated-measures z iteracjami (kluczowa decyzja)

Zamiast jednej odpowiedzi na pytanie zbieramy **K próbek** (iteracji) na to samo pytanie. Pojedyncza odpowiedź daje punkt; K próbek daje **rozkład**, a rozkład pozwala odróżnić:
- model **nie umie** pytania (myli się za każdym razem, często tak samo) → błąd *systematyczny*
- model **umie, ale jest niestabilny** (czasem trafia, czasem nie) → błąd *stochastyczny*

Zestaw pytań jest **zamrożony**: te same `qid` używane w każdej condition i każdej iteracji (paired / matched design). To odblokowuje testy parowane (McNemar, Cochran's Q, paired bootstrap) → więcej mocy statystycznej przy mniejszej liczbie pytań.

**Iteracje mają sens tylko przy temp>0** — przy temp=0 dekodowanie jest greedy, kolejne próbki są identyczne. Dlatego conditions deterministyczne dostają K=1 automatycznie (wymusza to `sweep.py`).

## Lokalizacja i środowisko

- **Katalog projektu**: `c:\Users\Andrzej\Desktop\aws\nlp-error-analysis\`
- **Python**: 3.12, venv w `.venv\`
- **Ollama**: 0.30.6 zainstalowane, modele na `D:\ollama_models\` (env var `OLLAMA_MODELS` w User scope)
- **Model**: `llama3.1:8b` (~5GB, pulled)
- **GPU**: dostępne
- **Search backend**: DuckDuckGo (free, no API key) przez `duckduckgo-search`

### Aktywacja środowiska (każda nowa sesja terminala)

```powershell
cd c:\Users\Andrzej\Desktop\aws\nlp-error-analysis
.\.venv\Scripts\Activate.ps1   # albo używaj .\.venv\Scripts\python.exe wprost
```

### Weryfikacja Ollama

```powershell
ollama list                                   # powinien być llama3.1:8b
Invoke-WebRequest http://localhost:11434/api/tags  # serwer odpowiada
```

Jeśli Ollama nie działa po restarcie systemu:
```powershell
$env:OLLAMA_MODELS = "D:\ollama_models"
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
```

## Pliki i co robią

| Plik | Rola |
|---|---|
| `dataset.py` | Stratified sample z HotpotQA distractor (train). Domyślnie **100 pytań** (25 × 4 buckety: easy/hard × comparison/bridge). Output: `data/samples_100.json` + `data/samples_100.manifest.json` (hash zestawu + qid). `question_set_hash()` — stabilny hash po posortowanych qid. CLI: `--n-per-bucket`, `--seed`, `--out`. |
| `prompts.py` | 3 szablony promptu (zero_shot, cot, verbose) + AGENT_SYSTEM_PROMPT dla trybu agent. |
| `search.py` | Wrapper na DuckDuckGo. `web_search(query, max_results=4)` → lista `SearchResult(title, url, snippet)`, retry x2. |
| `inference.py` | `InferenceConfig` (z polem `iters`), `infer(question, context, config, seed=None)`, `CONDITIONS` (7). Tryby `context` i `agent` (ReAct). Seed przekazywany per iterację → reprodukowalna losowość. |
| `sweep.py` | Batch runner, **potrójna pętla** condition × qid × iter. Zapis: `runs/<condition>/<qid>.json` z listą `samples`. Resume po liczbie **udanych** próbek (błędne powtarzane). Guard na hash zestawu. CLI: `--condition`, `--limit`, `--iters`, `--samples`, `--no-resume`. |
| `metrics.py` | `exact_match()`, `f1_score()`, `normalize_answer()` zgodnie z HotpotQA (Yang et al. 2018). |
| `judge.py` | LLM-as-Judge: drugie zapytanie do Llamy oceniające semantyczną równoważność. |
| `consistency.py` | **(nowy)** Metryki rozkładu: `majority`, `shannon_entropy`, `pass_at_1`, `pass_at_k` (nieobciążony estymator), `wrong_consistency`. Czyste funkcje, bez I/O. |
| `evaluate_all.py` | Zbiera wszystkie próbki, liczy EM/F1/Judge (Judge **dedupowany** po treści odpowiedzi w obrębie pytania). Output: `analysis/raw_long.csv` (per próbka) + `analysis/per_question.csv` (agregaty). |
| `smoke_test.py` | Sanity check: Ollama + model + 1 inferencja + 1 real sample. |
| `requirements.txt` | datasets, openai, pydantic, matplotlib, pandas, seaborn, scikit-learn, duckduckgo-search. |

## 7 conditions w `inference.py`

| Nazwa | Prompt | Temp | Mode | K (iters) | Co testuje |
|---|---|:-:|---|:-:|---|
| A_zero_t0 | zero_shot | 0.0 | context | 1 | baseline deterministic |
| B_zero_t07 | zero_shot | 0.7 | context | 10 | wpływ temperatury + rozkład |
| C_zero_t10 | zero_shot | 1.0 | context | 10 | wysoka temperatura + rozkład |
| D_cot_t0 | cot | 0.0 | context | 1 | wpływ Chain-of-Thought |
| E_cot_t07 | cot | 0.7 | context | 10 | CoT + losowość |
| F_verbose_t0 | verbose | 0.0 | context | 1 | explicit instruction format |
| G_agent_search | agent | 0.0 | agent | 4 | model szuka w DDG; K>1 bo search jest zmienny |

Łącznie: 3×100×1 + 3×100×10 + 1×100×4 = **3700 inferencji**.

## Schemat zapisu `runs/<condition>/<qid>.json`

```json
{ "qid": "...", "config": "B_zero_t07", "model": "llama3.1:8b",
  "temperature": 0.7, "prompt_template": "zero_shot", "mode": "context",
  "question": "...", "gold_answer": "...", "q_type": "bridge", "level": "hard",
  "n_iters_target": 10,
  "samples": [
    {"iter": 0, "seed": 1000, "answer": "...", "raw_output": "...",
     "latency_s": 1.2, "output_tokens": 14, "error": null, "agent_trajectory": []},
    {"iter": 1, "seed": 1001, "answer": "...", ...}
  ] }
```
Seed = `1000 + iter` → cały bieg reprodukowalny.

## Workflow uruchomienia (od zera)

```powershell
# 1. Generuje zamrożony zestaw 100 pytań + manifest z hashem (jednorazowo)
.\.venv\Scripts\python.exe dataset.py

# 2. Sanity check Ollama + llama3.1:8b
.\.venv\Scripts\python.exe smoke_test.py

# 3. Dry-run (3 pytania per condition, sprawdza potrójną pętlę)
.\.venv\Scripts\python.exe sweep.py --limit 3

# 4. Pełny sweep (~3700 inferencji). Resume działa, można przerywać.
.\.venv\Scripts\python.exe sweep.py

# 5. Ewaluacja: EM/F1/Judge per próbka + agregacja per pytanie
.\.venv\Scripts\python.exe evaluate_all.py
# Output: analysis/raw_long.csv (per próbka) + analysis/per_question.csv (agregaty)
```

### Kolumny `analysis/per_question.csv` (serce analizy)

| Kolumna | Znaczenie |
|---|---|
| `n_samples`, `n_error` | liczba udanych / błędnych próbek |
| `n_em_correct`, `em_pass1`, `em_passany` | trafność EM: liczba, pass@1 (=n/K), czy *jakakolwiek* poprawna |
| `n_judge_correct`, `judge_pass1`, `judge_passany` | to samo dla Judge |
| `majority_answer`, `majority_em`, `majority_judge` | self-consistency vote i jego poprawność |
| `agreement` | udział modalnej odpowiedzi (pewność modelu, proxy bez logitów) |
| `answer_entropy` | znormalizowana entropia Shannona rozkładu [0,1] |
| `n_distinct` | liczba unikalnych odpowiedzi |
| `wrong_consistency` | zgoda *wśród błędnych* próbek (systematyczny vs losowy błąd) |

## Aktualny stan

- ✅ Faza 1–3: dataset, prompts, inference, sweep, metrics, judge, evaluate
- ✅ **Redesign na iteracje + paired design** — dataset.py (zamrożony zestaw + hash), inference.py (seed per iter, pole `iters`), sweep.py (potrójna pętla, guard, resume po udanych próbkach), consistency.py (nowy), evaluate_all.py (dedup Judge, dwa CSV). Logika przetestowana jednostkowo.
- 🟡 **Pełny sweep do puszczenia** na nowym zestawie 100 pytań (stary `runs/` z 300-pytaniowego designu jest nieaktualny — evaluate_all czyta stary schemat wstecznie, ale nowy bieg robimy na samples_100).

## Kluczowe decyzje projektowe (do utrzymania)

- **Reading comprehension z podanym kontekstem** zamiast pure-agent — prowadzący sugerował kontekst w prompcie + EM/F1 działa z gold answers
- **7. condition `G_agent_search` z DuckDuckGo** — dodaje wymiar "search"
- **3 niezależne metryki** (EM, F1, Judge) — każda łapie inny błąd
- **Iteracje + paired design** — punkt → rozkład; matched qid → testy parowane; rozróżnienie błąd systematyczny vs stochastyczny (PB5)
- **100 pytań × K** zamiast 300×1 — głębia (rozkład per pytanie) potrzebna do self-consistency / kalibracji; 25/cell to akceptowalne minimum stratyfikacji
- **Judge dedupowany** po treści odpowiedzi — przy temp 0.7 model często powtarza odpowiedź, więc realnych wywołań sędziego dużo mniej niż 3700
- **Llama 3.1 8B jako i agent i sędzia** — minimalizuje zależność od API, open-weight

## Pułapki i co watchować

- **Sleep / hibernate Windows** zatrzymuje Python — sweep się przerywa, ale resume działa (liczy udane próbki)
- **Guard na hash zestawu** — jeśli przegenerujesz `samples_100.json` z innym seedem w trakcie biegu, sweep przerwie z błędem zamiast cicho mieszać dwa zestawy. Użyj `--no-resume` żeby zacząć od zera.
- **Ollama service** musi chodzić podczas sweepu
- **DuckDuckGo rate-limity** — przy agent mode czasem puste wyniki; dla G część wariancji wewnątrz-pytaniowej to zmienność searcha, nie samplingu modelu (odnotować w limitacjach)
- **duckduckgo-search deprecation warning** (rebranding na `ddgs`) — działa
- **Pakiety w `.venv\Scripts\python.exe`** — VSCode może pokazywać "package not installed" jeśli interpreter ustawiony na innego Pythona; faktyczne wykonanie działa
