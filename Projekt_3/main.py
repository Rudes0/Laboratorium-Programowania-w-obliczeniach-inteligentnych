import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# ==================================================
# 1. PARAMETRY PROGRAMU
# ==================================================

FOLDER_PROJEKTU = Path(__file__).resolve().parent

FOLDER_ZDJEC = FOLDER_PROJEKTU / "tekstury"
FOLDER_PROBEK = FOLDER_PROJEKTU / "probki"

PLIK_CSV = FOLDER_PROJEKTU / "cechy_tekstur.csv"

# Rozmiar wycinanej probki
ROZMIAR_PROBKI = (128, 128)

# Rozmiar wczytywanych obrazow
# None oznacza zachowanie oryginalnego rozmiaru
ROZMIAR_OBRAZU = (3024, 4032)

# Liczba poziomow jasnosci
LICZBA_POZIOMOW = 64

# Odleglosci pikseli
ODLEGLOSCI = [1, 3, 5]

# Kierunki w stopniach
KATY_STOPNIE = [0, 45, 90, 135]

# Kierunki w radianach
KATY = np.deg2rad(KATY_STOPNIE)

MAX_PROBEK_NA_ZDJECIE = 50

# Cechy GLCM
CECHY = [
    "dissimilarity",
    "correlation",
    "contrast",
    "energy",
    "homogeneity",
    "ASM"
]

# Rozszerzenia plikow obrazow
ROZSZERZENIA = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ==================================================
# 2. WCZYTYWANIE I WYCINANIE PROBEK
# ==================================================

def wycinaj_probki(
    folder_zdjec,
    folder_probek,
    rozmiar_probki=(128, 128),
    rozmiar_obrazu=None
):

    print("\n===================================")
    print("ETAP 1 - WYCINANIE PROBEK TEKSTUR")
    print("===================================")

    folder_zdjec = Path(folder_zdjec)
    folder_probek = Path(folder_probek)

    folder_probek.mkdir(
        parents=True,
        exist_ok=True
    )

    szerokosc_probki, wysokosc_probki = rozmiar_probki

    liczba_zdjec = 0
    liczba_probek = 0

    rng = np.random.default_rng(42)

    for folder_kategorii in sorted(folder_zdjec.iterdir()):

        if not folder_kategorii.is_dir():
            continue

        kategoria = folder_kategorii.name

        folder_wynikowy = folder_probek / kategoria

        folder_wynikowy.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"\nKategoria: {kategoria}")

        # Usuwanie starych wygenerowanych probek
        for stara_probka in folder_wynikowy.glob("*.png"):
            stara_probka.unlink()

        for sciezka in sorted(folder_kategorii.iterdir()):

            if sciezka.suffix.lower() not in ROZSZERZENIA:
                continue

            if not sciezka.is_file():
                continue

            with Image.open(sciezka) as obraz:

                obraz = obraz.convert("RGB")

                if rozmiar_obrazu is not None:
                    obraz = obraz.resize(
                        rozmiar_obrazu,
                        Image.Resampling.LANCZOS
                    )

                szerokosc, wysokosc = obraz.size

                liczba_zdjec += 1

                # Wszystkie mozliwe pozycje probek
                pozycje = []

                for y in range(
                    0,
                    wysokosc - wysokosc_probki + 1,
                    wysokosc_probki
                ):

                    for x in range(
                        0,
                        szerokosc - szerokosc_probki + 1,
                        szerokosc_probki
                    ):

                        pozycje.append((x, y))

                # Losowanie maksymalnie 50 probek
                liczba_do_wyciecia = min(
                    MAX_PROBEK_NA_ZDJECIE,
                    len(pozycje)
                )

                indeksy = rng.choice(
                    len(pozycje),
                    size=liczba_do_wyciecia,
                    replace=False
                )

                licznik = 0

                for indeks in indeksy:

                    x, y = pozycje[indeks]

                    probka = obraz.crop((
                        x,
                        y,
                        x + szerokosc_probki,
                        y + wysokosc_probki
                    ))

                    nazwa_probki = (
                        f"{sciezka.stem}"
                        f"__probka_{licznik:04d}.png"
                    )

                    probka.save(
                        folder_wynikowy / nazwa_probki
                    )

                    licznik += 1
                    liczba_probek += 1

                print(
                    f"Zdjecie: {sciezka.name} "
                    f"-> {licznik} probek"
                )

    print("\nPodsumowanie:")
    print("Liczba zdjec:", liczba_zdjec)
    print("Liczba probek:", liczba_probek)

    return liczba_probek

# ==================================================
# 3. OBLICZANIE CECH GLCM
# ==================================================

def oblicz_cechy_glcm(sciezka_probki):

    # Wczytanie obrazu w skali szarosci
    with Image.open(sciezka_probki) as obraz:
        obraz = obraz.convert("L")
        obraz = np.array(obraz)

    # Zmniejszenie liczby poziomow jasnosci
    # 256 poziomow -> 64 poziomy
    obraz = (
        obraz.astype(np.uint16)
        * LICZBA_POZIOMOW
        // 256
    ).astype(np.uint8)

    # Obliczenie macierzy GLCM
    glcm = graycomatrix(
        obraz,
        distances=ODLEGLOSCI,
        angles=KATY,
        levels=LICZBA_POZIOMOW,
        symmetric=True,
        normed=True
    )

    wyniki = {}

    # Obliczenie wszystkich cech
    for cecha in CECHY:

        wartosci = graycoprops(
            glcm,
            cecha
        )

        # Osobna cecha dla kazdej odleglosci
        # i kazdego kierunku
        for i, odleglosc in enumerate(ODLEGLOSCI):

            for j, kat in enumerate(KATY_STOPNIE):

                nazwa = (
                    f"{cecha}"
                    f"_d{odleglosc}"
                    f"_kat{kat}"
                )

                wyniki[nazwa] = float(
                    wartosci[i, j]
                )

    return wyniki


# ==================================================
# 4. GENEROWANIE ZBIORU CECH
# ==================================================

def generuj_zbior_cech(folder_probek, plik_csv):

    print("\n===================================")
    print("ETAP 2 - OBLICZANIE CECH GLCM")
    print("===================================")

    folder_probek = Path(folder_probek)

    dane = []

    for folder_kategorii in sorted(folder_probek.iterdir()):

        if not folder_kategorii.is_dir():
            continue

        kategoria = folder_kategorii.name

        print(f"\nKategoria: {kategoria}")

        licznik = 0

        for sciezka in sorted(folder_kategorii.glob("*.png")):

            cechy = oblicz_cechy_glcm(sciezka)

            # Nazwa kategorii tekstury
            cechy["kategoria"] = kategoria

            # Nazwa oryginalnego zdjecia
            # Potrzebna do poprawnego podzialu
            # na trening i test
            cechy["zdjecie"] = (
                f"{kategoria}/"
                f"{sciezka.name.split('__probka_')[0]}"
            )

            cechy["probka"] = sciezka.name

            dane.append(cechy)

            licznik += 1

        print("Liczba probek:", licznik)

    if not dane:
        raise ValueError(
            "Nie znaleziono probek tekstur."
        )

    df = pd.DataFrame(dane)

    # Zapis wektorow cech do CSV
    df.to_csv(
        plik_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nZapisano plik:", plik_csv)

    print("Liczba wektorow:", len(df))

    print("Liczba cech GLCM:", len(CECHY) * len(ODLEGLOSCI) * len(KATY))

    return df


# ==================================================
# 5. KLASYFIKACJA TEKSTUR - SVM
# ==================================================

def klasyfikuj_tekstury(plik_csv):

    print("\n===================================")
    print("ETAP 3 - KLASYFIKACJA TEKSTUR")
    print("===================================")

    # Wczytanie danych z pliku CSV
    df = pd.read_csv(plik_csv)

    # Wybranie kolumn zawierajacych cechy GLCM
    kolumny_cech = [
        kolumna
        for kolumna in df.columns
        if kolumna not in [
            "kategoria",
            "zdjecie",
            "probka"
        ]
    ]

    X = df[kolumny_cech]
    y = df["kategoria"]

    # Sprawdzenie liczby kategorii
    if y.nunique() < 2:
        raise ValueError(
            "Do klasyfikacji potrzebne sa "
            "co najmniej 2 kategorie tekstur."
        )

    # ==============================================
    # PODZIAL NA ZBIOR TRENINGOWY I TESTOWY
    # ==============================================

    rng = np.random.default_rng(42)

    indeksy_treningowe = []
    indeksy_testowe = []

    # Podzial wykonywany osobno dla kazdej kategorii
    for kategoria in sorted(df["kategoria"].unique()):

        dane_kategorii = df[
            df["kategoria"] == kategoria
        ]

        # Pobranie nazw oryginalnych zdjec
        # i konwersja do tablicy NumPy
        zdjecia = (
            dane_kategorii["zdjecie"]
            .unique()
            .to_numpy(copy=True)
        )

        # Potrzebujemy co najmniej dwoch zdjec
        if len(zdjecia) < 2:
            raise ValueError(
                f"Kategoria '{kategoria}' ma mniej niz "
                "2 niezalezne zdjecia. "
                "Potrzebne sa minimum 2 zdjecia."
            )

        # Losowe przemieszanie zdjec
        rng.shuffle(zdjecia)

        # 25% zdjec przeznaczamy do testowania
        liczba_testowych = max(
            1,
            int(round(len(zdjecia) * 0.25))
        )

        # Co najmniej jedno zdjecie musi zostac
        # w zbiorze treningowym
        liczba_testowych = min(
            liczba_testowych,
            len(zdjecia) - 1
        )

        zdjecia_testowe = zdjecia[:liczba_testowych]
        zdjecia_treningowe = zdjecia[liczba_testowych:]

        # Pobranie indeksow probek treningowych
        indeksy_treningowe.extend(
            dane_kategorii[
                dane_kategorii["zdjecie"].isin(
                    zdjecia_treningowe
                )
            ].index.tolist()
        )

        # Pobranie indeksow probek testowych
        indeksy_testowe.extend(
            dane_kategorii[
                dane_kategorii["zdjecie"].isin(
                    zdjecia_testowe
                )
            ].index.tolist()
        )

        print(f"\nKategoria: {kategoria}")
        print("Liczba zdjec:", len(zdjecia))
        print(
            "Zdjecia treningowe:",
            len(zdjecia_treningowe)
        )
        print(
            "Zdjecia testowe:",
            len(zdjecia_testowe)
        )

    # ==============================================
    # PRZYGOTOWANIE DANYCH
    # ==============================================

    X_train = X.iloc[indeksy_treningowe]
    X_test = X.iloc[indeksy_testowe]

    y_train = y.iloc[indeksy_treningowe]
    y_test = y.iloc[indeksy_testowe]

    print("\n===================================")
    print("PODSUMOWANIE PODZIALU DANYCH")
    print("===================================")

    print("Liczba probek treningowych:", len(X_train))
    print("Liczba probek testowych:", len(X_test))

    print("\nKategorie w zbiorze treningowym:")
    print(y_train.value_counts())

    print("\nKategorie w zbiorze testowym:")
    print(y_test.value_counts())

    # ==============================================
    # UTWORZENIE KLASYFIKATORA SVM
    # ==============================================

    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale"
        )
    )

    # Trenowanie modelu
    print("\nRozpoczynam trenowanie SVM...")

    model.fit(X_train, y_train)

    # ==============================================
    # TESTOWANIE KLASYFIKATORA
    # ==============================================

    przewidywania = model.predict(X_test)

    # Obliczenie dokladnosci
    dokladnosc = accuracy_score(
        y_test,
        przewidywania
    )

    print("\n===================================")
    print("WYNIKI KLASYFIKACJI")
    print("===================================")

    print(
        f"Dokladnosc klasyfikatora: "
        f"{dokladnosc * 100:.2f}%"
    )

    # Szczegolowy raport
    print("\nRaport klasyfikacji:")

    print(
        classification_report(
            y_test,
            przewidywania,
            zero_division=0
        )
    )

    # ==============================================
    # MACIERZ POMYLEK
    # ==============================================

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        przewidywania,
        cmap="Blues"
    )

    plt.title(
        "Macierz pomylek - klasyfikacja tekstur SVM"
    )

    plt.tight_layout()
    plt.show()

    return dokladnosc

# ==================================================
# 6. PROGRAM GLOWNY
# ==================================================

if __name__ == "__main__":

    print("===================================")
    print("ANALIZA I KLASYFIKACJA TEKSTUR")
    print("===================================")

    # ETAP 1
    # Wczytywanie zdjec i wycinanie probek

    liczba_probek = wycinaj_probki(
        FOLDER_ZDJEC,
        FOLDER_PROBEK,
        rozmiar_probki=ROZMIAR_PROBKI,
        rozmiar_obrazu=ROZMIAR_OBRAZU
    )

    # ETAP 2
    # Obliczanie cech GLCM

    dane = generuj_zbior_cech(
        FOLDER_PROBEK,
        PLIK_CSV
    )

    # ETAP 3
    # Klasyfikacja SVM

    dokladnosc = klasyfikuj_tekstury(
        PLIK_CSV
    )

    print("\n===================================")
    print("KONIEC PROGRAMU")
    print("===================================")