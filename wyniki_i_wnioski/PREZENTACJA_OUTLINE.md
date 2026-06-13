# Konspekt prezentacji przed grupą — Analiza błędów Llama 3.1 8B na HotpotQA

> ~8 slajdów treści + tytuł + backup. Cel: ~10–12 min. Zasada: 1 slajd = 1 myśl = 1 wizual.
> Asset = pliki z `wyniki_i_wnioski/`. Liczbę z Etapu 7 (Cohen's kappa) wstaw na slajd 7.

---

## Slajd 0 — Tytuł
- **Tytuł:** „Gdzie poddaje się model 8B? Analiza błędów Llama 3.1 w multi-hop QA"
- Podtytuł: Llama 3.1 8B · HotpotQA · 7 konfiguracji · sędzia Gemma 2 9B
- Autorzy + kurs NLP
- **Mówisz (hook, 20 s):** „Sprawdziliśmy nie *ile* razy model się myli, ale *jak* się myli — i czy da się to naprawić."

---

## Slajd 1 — Problem i metodologia
- **Na slajdzie (diagram przepływu, nie bullet-lista):**
  `100 zamrożonych pytań → 7 konfiguracji → 3400 odpowiedzi (C2/C3/C5 ×10) → 3 metryki + sędzia AI`
- Pytania multi-hop: odpowiedź wymaga **dwóch skoków** (np. reżyser filmu → jego miejsce urodzenia)
- Zbiór: HotpotQA (wariant `distractor`), 100 pytań zbalansowanych (25× bridge/comparison × easy/hard), **zamrożony raz**
- Model lokalnie przez Ollama; 7 konfiguracji = temperatura (0/0.5/1.0) × prompt (standard/CoT/expert) × agent z Wikipedią
- **Mówisz (60 s):** wyjaśnij multi-hop na jednym przykładzie, podkreśl „zamrożony zbiór = uczciwe porównania", wymień 3 osie zmian (temperatura, prompt, RAG).

---

## Slajd 2 — Kryzys klasycznych metryk  ⭐ (asset: `performance_by_config.png`)
- **Liczby na slajdzie:**
  - Exact Match = **0%** we *wszystkich* 7 konfiguracjach
  - F1 = **3–17%**
  - Sędzia AI (Gemma 2 9B) = **69–77%** (konfiguracje kontekstowe)
- **Mówisz (90 s):** Llama „rozlewa" odpowiedź w całe zdania — gold to „Elizabeth Taylor", model pisze „Dame Elizabeth Rosemond Taylor was a British-American actress…". EM/F1 to karzą jako 0, choć fakt jest poprawny. Sędzia-LLM wyciąga sedno. **Wniosek: miary n-gramowe są bezużyteczne do oceny wnioskowania LLM.**
- *Puenta:* to nie znaczy, że model jest świetny — patrz dalej.

---

## Slajd 3 — Typologia błędów  ⭐ (asset: `error_distribution.png`)
- **Liczby:** Multi-hop reasoning **80.9%** · Entity confusion 8.2% · Format 5.6% · Halucynacja 4.5% · Negacja 0.8%
- **Mówisz (60 s):** Gdy już się myli, w 4 na 5 przypadków to **załamanie drugiego skoku** — model poprawnie znajduje pierwszy fakt, ale nie łączy go z drugim. Halucynacje są **rzadkie** (4.5%) — model raczej „nie umie połączyć", niż „zmyśla". To kontruje popularny obraz LLM jako głównie halucynujących.

---

## Slajd 4 — Błędy systematyczne vs stochastyczne  ⭐ (asset: `systematic_stochastic.png`)
- **Na slajdzie (odczyt z wykresu):**
  - temp 0 (C1, C4, C6) → błędy **systematyczne** (0 stochastycznych — determinizm)
  - temp 0.5–1.0 (C2, C3, C5) → błędy **stochastyczne** rosną z temperaturą (C3 temp 1.0: 62/100 stochastycznych)
- **Mówisz (75 s):** Przy wyższej temperaturze model na to samo pytanie raz trafia, raz nie — błąd „migocze". To znaczy, że **część wiedzy tam jest, tylko niestabilna**. Stąd prosta recepta: *self-consistency* — zapytaj 10 razy, weź większość. Błędy systematyczne (te same w 100% prób) tego nie naprawisz — to realne luki.
- *To jest teza całości:* różne ustawienia zmieniają nie tyle ILE, co JAK (i czy naprawialnie) model się myli.

---

## Slajd 5 — Paradoks RAG-a: agent z Wikipedią wypada GORZEJ  ⭐ (asset: `performance_by_config.png`, podświetl C7)
- **Liczba:** C7 (agent + wyszukiwanie Wikipedii) = **~28%** sędziego vs 69–77% przy kontekście w prompcie
- Dodatkowo: błędy C7 są **systematyczne** (72/100) — powtarzalnie złe trajektorie
- **Mówisz (60 s):** Kontrintuicyjne: daliśmy modelowi narzędzie do szukania faktów i było *gorzej*. Powód: model 8B gubi się w pętli ReAct — kiepskie zapytania, złe wyszukania, błądzenie w 3 krokach. **Wniosek: dla małych modeli „podaj kontekst" bije „pozwól szukać".** Mocny, dyskusyjny punkt na pytania z sali.

---

## Slajd 6 — Wpływ promptu (CoT / expert) — krótki (asset: `performance_by_config.png`)
- **Liczby:** C1 standard 0.73 · C4 CoT 0.75 · C6 expert **0.77** (najlepszy) · C3 temp 1.0 0.69 (najgorszy z kontekstowych)
- **Mówisz (45 s):** Chain-of-Thought i rozbudowany prompt dają **marginalny** wzrost (~+2–4 pkt), nie rewolucję — ale CoT zmienia *profil* błędów (więcej format error: F1 leci do 3% bo model jeszcze bardziej „leje wodę"). Wysoka temperatura lekko szkodzi.
- *(Slajd opcjonalny — jeśli ciśnie czas, scal z Slajdem 2.)*

---

## Slajd 7 — Walidacja ludzka: Cohen's kappa  ⭐ (asset: zrzut konsoli z Etapu 7)
- **Na slajdzie:** zrzut wyniku `07_calculate_kappa.py` — Accuracy __% i **Cohen's kappa = __** (wstaw po adnotacji)
- Kontekst: 100 błędów ocenione **niezależnie przez człowieka**, porównane z ukrytą oceną Gemmy
- **Uwaga metodologiczna (na slajdzie, mały tekst):** próbka skośna (84% multi-hop) → kappa może być umiarkowana mimo wysokiej zgodności % (paradoks kappy) — to *zaleta* prezentacji, nie wada
- **Case study (mówisz, 45 s):** Gemma czasem daje się nabrać na „lanie wody" o zbliżonej strukturze zdania i zalicza złą odpowiedź. Dlatego walidacja człowiekiem jest potrzebna — **pokazujemy, że pipeline AI ma margines błędu i wymaga punktowego nadzoru.**

---

## Slajd 8 — Wnioski + future work
- **3 findingi (duże, zapamiętywalne):**
  1. Miary n-gramowe (EM/F1) są martwe dla oceny LLM — liczy się sędzia semantyczny
  2. Bariera modeli 8B to **drugi skok rozumowania** (80.9% błędów), nie halucynacje
  3. RAG bez silnego modelu szkodzi; self-consistency naprawia błędy stochastyczne, nie systematyczne
- **Ograniczenia:** 100 pytań (25/komórkę), jeden anotator, sędzia 9B (margines błędu — stąd kappa), HotpotQA możliwe w pretreningu
- **Future work:** większy sędzia (GPT-4o/Gemini), self-consistency jako mechanizm produkcyjny, lepszy scaffolding agenta
- **Mówisz (45 s):** zamknij tezą ze Slajdu 4.

---

## Slajdy BACKUP (za końcem — na pytania z sali)
- B1: Pełna tabela 7×3 (EM/F1/judge per konfiguracja) z dokładnymi liczbami
- B2: Definicje 5 kategorii błędów + po 1 przykładzie (z `INSTRUKCJA_ADNOTACJI.md`)
- B3: Konkretny przykład multi-hop failure (pytanie → zła odp → dlaczego) — np. Sid Avery / Elizabeth Taylor z próbki
- B4: Pełny rozkład systematyczny/stochastyczny per konfiguracja (liczby z wykresu 3)
- B5: Sanity-check EM=0 (czy to realny efekt rozwlekłości, czy artefakt — jedna odpowiedź na czepliwe pytanie)

---

## Mapowanie na pytania badawcze (gdyby prowadzący pytał o strukturę)
- PB „jak grupują się błędy" → Slajd 3
- PB „wpływ konfiguracji" → Slajdy 4, 5, 6
- PB „zgodność metryk / wiarygodność oceny" → Slajdy 2, 7
- PB „systematyczne vs stochastyczne" → Slajd 4

## Timing (cel ~11 min)
| Slajd | min |
|---|:-:|
| 0 Tytuł | 0.5 |
| 1 Metodologia | 1.0 |
| 2 Kryzys metryk | 1.5 |
| 3 Typologia | 1.0 |
| 4 Sys/stoch | 1.5 |
| 5 RAG paradoks | 1.0 |
| 6 Prompt (opc.) | 0.75 |
| 7 Kappa + case study | 1.5 |
| 8 Wnioski | 1.0 |
| Bufor / pytania | reszta |
