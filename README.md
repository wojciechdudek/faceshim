# faceshim

Rozpoznawanie twarzy dla Double Take w **jednym kontenerze**: InsightFace (SCRFD + ArcFace `buffalo_l`)
na `onnxruntime`, wystawione przez API zgodne z DeepStackiem. Double Take widzi "deepstacka" i nie
wymaga żadnych zmian poza `url`. Zastępuje CompreFace (5 kontenerów, TensorFlow, Java, Postgres)
przy ~600 MB RAM, modelu ładowanym raz na starcie i pełnej rozdzielczości wejścia.

```
Frigate ──MQTT──▶ Double Take ──HTTP (DeepStack API)──▶ faceshim
                        │                                  │
                        └── sensor.double_take_<osoba> ◀───┘ (jak dotąd)
```

## API

| metoda | ścieżka | pola | odpowiedź |
|---|---|---|---|
| `GET` | `/` | – | `{"success":true,"model":"buffalo_l","users":[...],"embeddings":n}` |
| `POST` | `/v1/vision/face/register` | `image`, `userid` | `{"success":true,"message":"face added",...}` |
| `POST` | `/v1/vision/face/recognize` | `image`[, `min_confidence`] | `{"success":true,"predictions":[{"userid","confidence","x_min","y_min","x_max","y_max","similarity","candidate","det_score"}],"duration":ms}` |
| `GET/POST` | `/v1/vision/face/list` | – | `{"success":true,"faces":[...]}` |
| `POST` | `/v1/vision/face/delete` | `userid` | `{"success":true}` |

`confidence` (0–1) to skalibrowane podobieństwo, którego używa Double Take. `similarity` to surowy
cosinus z ArcFace, `candidate` to najlepszy kandydat nawet gdy wynik spadł poniżej progu — oba służą
wyłącznie do strojenia (Double Take je ignoruje).

## Zmienne środowiskowe

| zmienna | domyślnie | znaczenie |
|---|---|---|
| `SIM_LOW` | `0.30` | podobieństwo kosinusowe odpowiadające 0 % |
| `SIM_HIGH` | `0.70` | podobieństwo kosinusowe odpowiadające 100 % |
| `MIN_CONFIDENCE` | `0.20` | poniżej tej pewności twarz wraca jako `unknown` |
| `MIN_FACE_PX` | `40` | mniejsze twarze (bok boxa w px) są pomijane |
| `PAD_RETRY_BELOW` | `0.70` | przy rozpoznawaniu: gdy detektor nie znajdzie twarzy albo jest jej mniej pewny niż to, obraz dostaje margines i detekcja jest powtarzana; przy rejestracji margines jest próbowany zawsze i zostaje lepsza detekcja — ratuje ciasno kadrowane portrety (głowa na cały kadr) |
| `PAD_FRACTION` | `0.5` | szerokość tego marginesu jako ułamek dłuższego boku; `0` wyłącza ponowną detekcję |
| `MAX_FACES` | `5` | ile twarzy z jednego obrazu zwracać (największe najpierw) |
| `DET_SIZE` | `640` | rozmiar wejścia detektora; twarz do rozpoznania jest wycinana z oryginału |
| `MODEL_NAME` | `buffalo_l` | pakiet modeli InsightFace zaszyty w obrazie (`buffalo_s` = lżejszy, mniej trafny) |
| `ORT_PROVIDERS` | `CPUExecutionProvider` | np. `OpenVINOExecutionProvider,CPUExecutionProvider` na iGPU Intela |
| `OMP_NUM_THREADS` | `2` | wątki onnxruntime |
| `DATA_DIR` | `/data` | tu leży `faces.json` z wzorcami |

## Build i publikacja

```bash
make build-local        # obraz na lokalną architekturę, tag :dev
make run                # uruchamia go na :5002 (5000 na macOS zajmuje AirPlay) z wolumenem faceshim-data
make selfcheck          # register -> recognize -> delete na zdjęciu testowym insightface

make build IMAGE=ghcr.io/<owner>/faceshim TAG=0.1.5   # buildx linux/amd64 + push
```

Albo tag `vX.Y.Z` w repo — workflow `.github/workflows/publish.yml` zbuduje `linux/amd64`
i wypchnie do `ghcr.io/<owner>/faceshim:X.Y.Z` oraz `:latest`. Modele są pobierane **w trakcie
builda** i zaszywane w obrazie; kontener w runtime nie sięga do sieci.

Wariant na iGPU Intela: `docker buildx build --build-arg ORT_PACKAGE=onnxruntime-openvino==1.29.0 ...`,
w compose `devices: [/dev/dri:/dev/dri]` i `ORT_PROVIDERS=OpenVINOExecutionProvider,CPUExecutionProvider`.

## Wpięcie w stack

1. `compose.example.yml` → do compose obok Double Take. `/data` to katalog na hoście, jak przy
   pozostałych usługach — właściciel jest poprawiany automatycznie przy starcie.
2. `doubletake.example.yml` → blok `detectors` w `config.yml` Double Take. **Zostaw CompreFace obok**
   na czas testów.
3. Double Take → zakładka **Train** → **Sync**: wypycha istniejące zdjęcia treningowe do faceshim
   (`register`). Zdjęcia bez wykrywalnej twarzy dostaną błąd `no face found` – usuń je.
4. Zakładka **Matches** → przycisk odświeżenia na 10 starych zdarzeniach: Double Take pokaże wynik
   CompreFace i faceshim obok siebie.

## Porównanie z CompreFace na wszystkich zapisanych zdarzeniach

`evaluate.py` czyta bazę SQLite Double Take'a i jego zapisane obrazy, przepuszcza każdy przez faceshim
i drukuje tabelę: co zapisał Double Take (np. `compreface wojtek 97.5`) obok wyniku faceshim, a na
końcu rozkłady surowego podobieństwa dla domowników i obcych oraz sugestię `SIM_LOW` / `SIM_HIGH`.
Na hoście, obok działającego kontenera:

```bash
docker run --rm --network container:faceshim -v /home/double-take:/dt:ro \
    --entrypoint python ghcr.io/wojciechdudek/faceshim:0.1.5 evaluate.py --storage /dt
```

`--network container:faceshim` współdzieli sieć z serwisem, więc nie trzeba znać nazwy sieci
Dockera ani wystawiać portu. `--limit N` ogranicza do N najnowszych wierszy.

## Kalibracja na własnej kamerze

`confidence` liczy się liniowo między `SIM_LOW` (0 %) i `SIM_HIGH` (100 %). Skala ArcFace zależy
od optyki i światła, więc te dwa punkty ustawia się raz, na własnych danych:

1. Przelicz przez faceshim ~10 zdarzeń z domownikami i ~5 z obcymi; w logu kontenera (albo w polu
   `similarity` odpowiedzi) zobaczysz surowe podobieństwa.
2. `SIM_HIGH` ustaw tak, żeby dobre, frontalne twarze domowników lądowały ≥ 0.9 po przeliczeniu;
   `SIM_LOW` tak, żeby obcy nie przekraczali 0.5. Typowo dla `buffalo_l`: domownicy 0.5–0.75,
   obcy ≤ 0.3.
3. Progu w automatyzacji furtki (np. `> 90`) **nie obniżaj** – strój `SIM_*`, nie konsumenta.

## Gdy coś nie działa

- **Kontener wychodzi z `is not writable by uid 1000`** — zamontowany `/data` jest tylko do
  odczytu, albo uruchomiłeś kontener z własnym `user:` na katalogu należącym do roota. Bez `user:`
  entrypoint sam poprawia właściciela katalogu przy starcie (kontener startuje jako root wyłącznie
  po to i zaraz zrzuca uprawnienia do `app`, uid 1000).
- **Wszystko wraca jako `unknown`** — sprawdź `GET /`: pole `embeddings` musi być > 0. Jeśli 0,
  Sync w Double Take nie doszedł (patrz wyżej) — powtórz go.
- **`no face found` przy rejestracji zdjęcia z wyraźną twarzą** — portret wypełniający kadr, z obciętą
  górą głowy; od 0.1.5 obsługiwane automatycznie (`PAD_RETRY_BELOW`). Zdjęcia zarejestrowane starszą
  wersją warto zarejestrować ponownie: usuń `faces.json`, zrestartuj, w Double Take zrób Sync.
- **Domownicy poniżej progu mimo wyraźnego `similarity`** — to kalibracja, nie rozpoznawanie;
  sekcja wyżej.

## Ograniczenia

- Brak uwierzytelniania – serwis jest przeznaczony do sieci Dockera / LAN, nie do wystawiania na świat.
- Jeden worker, inferencja szeregowana blokadą; przy kilku zdarzeniach na minutę to bez znaczenia.
- Wzorce w JSON – przy tysiącach embeddingów przeszukiwanie liniowe zacznie być wolne; dla domu to
  setki, nie tysiące.
- Zmiana `MODEL_NAME` unieważnia zapisane embeddingi (inna przestrzeń wektorów) – po zmianie usuń
  `faces.json` i zrób Sync jeszcze raz.
