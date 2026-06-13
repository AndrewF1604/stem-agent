"""Czyści komórki oznaczone jako '4' (negacja) w adnotacja_reczna.csv, żeby
przejść je ponownie z poprawioną definicją. Robi backup przed zmianą.

Uruchom raz:
    python reset_negacja.py
Potem:
    python annotate_cli.py        # poprosi tylko o wyczyszczone wiersze
    python 07_calculate_kappa.py
"""
import csv
import shutil

csv.field_size_limit(10_000_000)
PATH = "data/manual_annotation/adnotacja_reczna.csv"

with open(PATH, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.reader(f))
header, data = rows[0], rows[1:]
hidx = len(header) - 1

to_clear = [r for r in data if r[hidx].strip() == "4"]
if not to_clear:
    print("Brak komórek oznaczonych '4' — nic do czyszczenia.")
else:
    shutil.copy(PATH, PATH.replace(".csv", ".bak.csv"))
    for r in to_clear:
        r[hidx] = ""
    with open(PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)
    print(f"Wyczyszczono {len(to_clear)} komórek (kat. 4).")
    print("Backup: data/manual_annotation/adnotacja_reczna.bak.csv")
    print("Teraz: python annotate_cli.py  (poprosi tylko o te wiersze)")
