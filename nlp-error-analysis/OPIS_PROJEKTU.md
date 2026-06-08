# Opis projektu — analiza błędów modelu językowego w pytaniach multi-hop

> Wysokopoziomowy opis: o co chodzi, jak to działa i na czym polega. Bez szczegółów technicznych.

## O co chodzi

Projekt bada **jakie błędy popełnia model językowy** (Llama 3.1 8B) odpowiadając na trudne pytania wymagające łączenia faktów. Nie pytamy tylko „ile razy się pomylił", lecz **jak konkretnie się myli** — czy zmyśla, czy gubi się w rozumowaniu, czy myli podobne do siebie obiekty — i czy rodzaj błędów zmienia się, gdy zmieniamy sposób zadawania pytań.

To klasyczne podejście z lingwistyki komputerowej zwane *error analysis*: zamiast jednej liczby (np. „70% poprawnych"), opisujemy **mapę błędów** — ich rodzaje, częstości i to, od czego zależą.

## Na czym polegają pytania (multi-hop QA)

Materiałem są pytania z bazy **HotpotQA**. Ich cechą jest to, że odpowiedź wymaga **kilku kroków rozumowania** — połączenia informacji z dwóch różnych źródeł. Przykładowo: „W którym mieście urodził się reżyser filmu X?" wymaga najpierw ustalenia reżysera, a dopiero potem miejsca jego urodzenia. Stąd nazwa *multi-hop* (wieloskokowe).

Do każdego pytania dołączony jest komplet tekstów (fragmentów encyklopedycznych): część zawiera potrzebne fakty, część to celowe rozpraszacze. Model dostaje pytanie razem z tymi tekstami i ma znaleźć odpowiedź — to test **rozumienia ze zrozumieniem**, nie wiedzy z pamięci.

Pytania różnią się dwoma cechami, które śledzimy:
- **typ**: *bridge* (łańcuch faktów A→B→C) lub *comparison* (porównanie dwóch obiektów),
- **trudność**: *easy* lub *hard*.

## Jak zbieramy dane — zamrożony zestaw pytań

Na początku **losujemy raz 100 pytań** w sposób zbalansowany (po równo z każdej kombinacji typu i trudności), **zapisujemy je do pliku** i dalej pracujemy **wyłącznie na tym jednym, niezmiennym zbiorze**. Nigdy nie losujemy ponownie. Dzięki temu wszystkie późniejsze porównania dotyczą dokładnie tych samych pytań — to fundament rzetelności całego badania.

## Warianty zadawania pytań (7 konfiguracji)

Te same 100 pytań zadajemy modelowi na **7 różnych sposobów**, żeby zobaczyć, co wpływa na błędy. Zmieniamy trzy rzeczy:

1. **Temperaturę** — parametr „odwagi" modelu. Niska (0) = odpowiedzi przewidywalne, zawsze takie same. Wysoka (1) = odpowiedzi bardziej losowe i kreatywne. Sprawdzamy trzy poziomy.
2. **Styl polecenia (prompt)** — od suchego „odpowiedz krótko", przez prośbę o **rozumowanie krok po kroku** (Chain-of-Thought), po wersję z rozbudowaną instrukcją.
3. **Tryb pracy** — albo model dostaje potrzebne teksty od razu w pytaniu, albo musi **sam wyszukać** informacje w internecie (przez wyszukiwarkę), zanim odpowie.

Każda konfiguracja to inny „charakter" modelu — i pytanie brzmi, czy popełnia wtedy inne rodzaje błędów.

## Dlaczego pytamy wielokrotnie (iteracje)

Gdy model pracuje z podwyższoną temperaturą, **na to samo pytanie może odpowiedzieć różnie za każdym razem**. Dlatego dla konfiguracji losowych zadajemy każde pytanie **wielokrotnie** (np. 10 razy). Zamiast jednej odpowiedzi otrzymujemy **rozkład odpowiedzi** — i to jest najcenniejsza część projektu, bo pozwala rozróżnić dwa zupełnie różne typy pomyłek:

- **błąd systematyczny** — model myli się **za każdym razem tak samo** (uparcie podaje tę samą złą odpowiedź). To znak, że czegoś nie wie albo ma silne, błędne przekonanie.
- **błąd stochastyczny** — model **czasem trafia, czasem nie**, a złe odpowiedzi są różne. To znak, że „w zasadzie potrafi", ale jest niestabilny.

To rozróżnienie ma praktyczne znaczenie: błędy stochastyczne można w dużej mierze „naprawić", pytając kilka razy i biorąc odpowiedź najczęstszą (technika *self-consistency*). Błędów systematycznych — nie.

Konfiguracje deterministyczne (temperatura 0) pytamy tylko raz, bo i tak dałyby identyczne odpowiedzi.

## Jak oceniamy poprawność (trzy metryki)

Ocena „czy odpowiedź jest dobra" nie jest oczywista, więc używamy **trzech niezależnych miar**, bo każda łapie co innego:

1. **Dokładne dopasowanie** — czy odpowiedź po uproszczeniu (małe litery, bez interpunkcji) jest identyczna ze wzorcową. Surowe, ale ślepe na parafrazy.
2. **Pokrycie słów (F1)** — jak bardzo odpowiedź pokrywa się słowami ze wzorcem. Łapie częściowe trafienia.
3. **Sędzia-model (LLM-as-Judge)** — drugi model ocenia, czy odpowiedź **znaczy to samo** co wzorcowa, tolerując parafrazy i inne sformułowania tej samej encji.

Porównanie tych trzech miar jest osobnym wynikiem: pokazuje, gdzie „sztywne" metryki niesprawiedliwie karzą dobre odpowiedzi (np. za format), a sędzia-model widzi sens.

## Klasyfikacja błędów (taksonomia)

Każdy błąd przypisujemy do jednej z **pięciu kategorii**:
- **Halucynacja** — odpowiedź nie wynika z dostarczonych tekstów,
- **Błąd rozumowania wieloskokowego** — pierwszy fakt znaleziony dobrze, zły wniosek w drugim kroku,
- **Błąd formatu** — model zna odpowiedź, ale podaje ją w złej postaci,
- **Błąd negacji** — przeoczenie „nie",
- **Pomylenie encji** — wybór powiązanego, ale błędnego obiektu.

Dodatkowo każdy błąd oznaczamy jako systematyczny albo stochastyczny (z analizy wielu iteracji).

Klasyfikacji dokonujemy dwutorowo: część błędów **ręcznie** (autor), a wszystkie **automatycznie** (przez model), po czym mierzymy **zgodność** obu metod — to walidacja, że automatyczna klasyfikacja jest wiarygodna.

## Co z tego wynika (pytania badawcze)

Całość odpowiada na pięć pytań:
1. Jaka jest **dystrybucja** rodzajów błędów?
2. Czy zależy od **typu i trudności** pytania?
3. Czy zmienia ją **konfiguracja** (temperatura, styl promptu, wyszukiwarka)?
4. Jak **zgodne** są trzy metryki oceny?
5. Czy błędy są raczej **systematyczne czy stochastyczne** — i ile naprawia pytanie wielokrotne?

## Pętla całości w skrócie

> Wylosuj raz 100 zbalansowanych pytań → zapisz do pliku → zadaj je modelowi na 7 sposobów (część wielokrotnie) → oceń każdą odpowiedź trzema metrykami → sklasyfikuj błędy → policz dystrybucje i zależności → opisz w raporcie i przedstaw na prezentacji.

Efektem końcowym jest **mapa błędów** modelu w zadaniu multi-hop QA: nie tylko ile się myli, ale **jak**, **kiedy** i **czy da się to naprawić**.
