"""Interaktywne narzędzie do ręcznej adnotacji błędów (Etap 6).

Pokazuje po kolei każdy błędny przypadek: pytanie, złotą odpowiedź, błędną
odpowiedź modelu oraz KLUCZOWE FAKTY z HotpotQA (zdania wskazane przez zbiór
jako wymagane do odpowiedzi — czyli łańcuch rozumowania). Na żądanie ('c')
pokazuje PEŁNY kontekst, który faktycznie widział model.

WAŻNE o kontekście: model w konfiguracjach C1–C6 dostawał CAŁY kontekst (wszystkie
pasaże), nie obcięty. Obcięcie w pliku CSV to tylko artefakt przygotowania
adnotacji — NIE oceniaj „model nie miał informacji" na podstawie tego, czego Ty
nie widzisz. Konfiguracja C7 (agent) w ogóle nie dostała tych tekstów — szukała
sama w Wikipedii, więc dla C7 kontekst z datasetu jest jedynie poglądowy.

Autozapis po każdym wpisie + wznawianie (pomija już wypełnione wiersze).

Uruchom z głównego folderu projektu:
    python annotate_cli.py

Sterowanie:
    1-5  przypisz kategorię błędu
    c    pokaż pełny kontekst, który widział model
    s    pomiń ten przypadek (wróci na końcu)
    b    cofnij się do poprzedniego wiersza
    q    zapisz i wyjdź

To zadanie jest z założenia LUDZKIE — Twoja niezależna ocena służy do walidacji
sędziego AI przez Cohen's kappa (Etap 7). Oceniaj samodzielnie.
"""
from __future__ import annotations

import csv
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

csv.field_size_limit(10_000_000)

CSV_PATH = os.path.join("data", "manual_annotation", "adnotacja_reczna.csv")
QUESTIONS_PATH = os.path.join("data", "sampled_100_questions.json")
ANSWER_LIMIT = 2000
CONTEXT_LIMIT = 1200

LEGEND = {
    "1": "Halucynacja — informacja zmyślona, nieobecna w tekście",
    "2": "Multi-hop reasoning — nie połączył 2 faktów / nie wykonał porównania "
         "(TU też: 'brak informacji'/odpowiedź przecząca, choć fakty były w tekście)",
    "3": "Format — znał fakt, ale zła forma (lanie wody zamiast krótkiej odp)",
    "4": "Negacja — model ŹLE ODCZYTAŁ negację w TEKŚCIE ŹRÓDŁOWYM "
         "(źródło: 'nie był' → model rozumie 'był'). NIE chodzi o to, że odpowiedź "
         "modelu zawiera 'nie'! (bardzo rzadkie)",
    "5": "Entity confusion — pomylił powiązaną encję (np. aktor drugoplanowy)",
}


def load() -> tuple[list[str], list[list[str]]]:
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def save(header: list[str], data: list[list[str]]) -> None:
    with open(CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def load_questions() -> dict:
    if not os.path.exists(QUESTIONS_PATH):
        return {}
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)}


def resolve_support(qrec: dict) -> list[tuple[str, str]]:
    """Zdania wskazane przez HotpotQA jako wymagane do odpowiedzi (łańcuch faktów)."""
    sf = qrec.get("supporting_facts") or {}
    ctx = qrec.get("context") or {}
    titles, sents = ctx.get("title", []), ctx.get("sentences", [])
    tidx = {t: i for i, t in enumerate(titles)}
    out = []
    for t, sid in zip(sf.get("title", []), sf.get("sent_id", [])):
        if t in tidx and 0 <= sid < len(sents[tidx[t]]):
            out.append((t, sents[tidx[t]][sid].strip()))
    return out


def format_full_context(qrec: dict) -> str:
    ctx = qrec.get("context") or {}
    blocks = [f"[{t}]\n{' '.join(ss)}" for t, ss in zip(ctx.get("title", []), ctx.get("sentences", []))]
    return "\n\n".join(blocks)


def print_full_context(qrec: dict | None, konfig: str, csv_ctx: str) -> None:
    print("\n" + "·" * 70)
    if konfig == "C7":
        print("UWAGA: C7 (agent) NIE dostał tych tekstów — szukał sam w Wikipedii.")
        print("Poniżej kontekst z datasetu tylko poglądowo.")
    if qrec:
        print("PEŁNY KONTEKST (to widział model):\n")
        print(format_full_context(qrec))
    else:
        print("PEŁNY KONTEKST (z pliku CSV — pytania nie znaleziono w źródle):\n")
        print(csv_ctx)
    print("·" * 70)


def show(row: list[str], qrec: dict | None, idx: int, total: int, done: int) -> None:
    konfig, pytanie, gold, bledna = row[1], row[2], row[3], row[4]
    print("\n" + "=" * 70)
    print(f"  Wiersz {idx + 1}/{total}   |   zaadnotowano: {done}/{total}   |   config: {konfig}")
    print("=" * 70)
    print(f"\nPYTANIE:\n  {pytanie}")
    print(f"\nZŁOTA ODPOWIEDŹ (prawda):\n  {gold}")
    ans = bledna if len(bledna) <= ANSWER_LIMIT else bledna[:ANSWER_LIMIT] + f" […obcięto; łącznie {len(bledna)} znaków — bardzo długa odpowiedź, sygnał błędu formatu/halucynacji]"
    print(f"\nODPOWIEDŹ MODELU (błędna):\n  {ans}")

    facts = resolve_support(qrec) if qrec else []
    if facts:
        print("\nKLUCZOWE FAKTY (HotpotQA — czego wymaga odpowiedź):")
        for t, s in facts:
            print(f"  • [{t}] {s}")
    if konfig == "C7":
        print("\n  (!) C7 to agent — kontekstu NIE dostał, szukał w Wikipedii.")

    print("\n" + "-" * 70)
    for k, v in LEGEND.items():
        print(f"  {k} = {v}")
    print("  c = pełny kontekst   s = pomiń   b = cofnij   q = zapisz i wyjdź")


def main() -> None:
    if not os.path.exists(CSV_PATH):
        print(f"Nie znaleziono {CSV_PATH}. Uruchom z głównego folderu projektu.")
        return

    header, data = load()
    questions = load_questions()
    if not questions:
        print("UWAGA: brak data/sampled_100_questions.json — kluczowe fakty i pełny "
              "kontekst niedostępne, lecę na skróconym kontekście z CSV.\n")
    hidx = len(header) - 1
    total = len(data)

    order = [i for i in range(total) if not data[i][hidx].strip()]
    if not order:
        print(f"Wszystkie {total} wierszy już zaadnotowane. Uruchom: python 07_calculate_kappa.py")
        return

    print(f"Do zaadnotowania: {len(order)} z {total} (reszta już wypełniona).")
    pos = 0
    while pos < len(order):
        i = order[pos]
        row = data[i]
        qrec = questions.get(row[0])
        done = sum(1 for r in data if r[hidx].strip())
        show(row, qrec, i, total, done)

        while True:
            ans = input("\nTwoja ocena (1-5 / c / s / b / q): ").strip().lower()
            if ans == "c":
                csv_ctx = row[5] if len(row) > 5 else ""
                print_full_context(qrec, row[1], csv_ctx)
                continue
            if ans == "q":
                save(header, data)
                done = sum(1 for r in data if r[hidx].strip())
                print(f"\nZapisano. Zaadnotowano {done}/{total}. Wznowisz później od niewypełnionych.")
                return
            if ans == "b":
                pos = max(0, pos - 1)
                break
            if ans == "s":
                pos += 1
                break
            if ans in LEGEND:
                data[i][hidx] = ans
                save(header, data)
                pos += 1
                break
            print("  ! Wpisz 1-5, albo c / s / b / q.")

    done = sum(1 for r in data if r[hidx].strip())
    save(header, data)
    print(f"\nZapisano. Zaadnotowano {done}/{total}.")
    if done == total:
        print("Komplet! Teraz uruchom: python 07_calculate_kappa.py")
    else:
        print("Możesz wrócić później — narzędzie wznowi od niewypełnionych.")


if __name__ == "__main__":
    main()
