import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import pyransac3d as pyrsc

from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN


# ==================================================
# PARAMETRY PROGRAMU
# ==================================================

LICZBA_ITERACJI = 1000
PROG_RANSAC = 0.05

EPS_DBSCAN = 1.5
MIN_SAMPLES_DBSCAN = 5

# Przesuniecia chmur w przestrzeni
PRZESUNIECIE_PIONOWEJ = [20, 0, 0]
PRZESUNIECIE_CYLINDRA = [40, 0, 0]


# ==================================================
# 1. WCZYTYWANIE PLIKOW XYZ
# ==================================================

def wczytaj_xyz(nazwa_pliku):

    punkty = []

    with open(nazwa_pliku, "r", encoding="utf-8") as plik:

        reader = csv.reader(plik, delimiter=" ")

        for wiersz in reader:

            wiersz = [x for x in wiersz if x]

            if len(wiersz) == 3:

                x = float(wiersz[0])
                y = float(wiersz[1])
                z = float(wiersz[2])

                punkty.append([x, y, z])

    return np.array(punkty, dtype=float).reshape(-1, 3)


# ==================================================
# 2. LACZENIE TRZECH CHMUR PUNKTOW
# ==================================================

def polacz_chmury():

    folder = Path(__file__).resolve().parent

    pozioma = wczytaj_xyz(folder / "pozioma.xyz")
    pionowa = wczytaj_xyz(folder / "pionowa.xyz")
    cylinder = wczytaj_xyz(folder / "cylinder.xyz")

    print("\nWCZYTANE PLIKI:")
    print("Pozioma:", len(pozioma), "punktow")
    print("Pionowa:", len(pionowa), "punktow")
    print("Cylinder:", len(cylinder), "punktow")

    if min(len(pozioma), len(pionowa), len(cylinder)) == 0:
        raise ValueError("Jeden z plikow XYZ jest pusty.")

    # Rozsuniecie powierzchni
    pionowa = pionowa + np.array(PRZESUNIECIE_PIONOWEJ)
    cylinder = cylinder + np.array(PRZESUNIECIE_CYLINDRA)

    # Polaczenie wszystkich punktow
    punkty = np.vstack((
        pozioma,
        pionowa,
        cylinder
    ))

    # Zapis do jednego pliku
    np.savetxt(
        folder / "chmura.xyz",
        punkty,
        fmt="%.6f"
    )

    print("\nUtworzono plik chmura.xyz")
    print("Laczna liczba punktow:", len(punkty))

    return punkty


# ==================================================
# 3. OBLICZANIE PLASZCZYZNY Z TRZECH PUNKTOW
# ==================================================

def oblicz_plaszczyzne(p1, p2, p3):

    v1 = p2 - p1
    v2 = p3 - p1

    normalny = np.cross(v1, v2)

    dlugosc = np.linalg.norm(normalny)

    if dlugosc < 1e-10:
        return None

    normalny = normalny / dlugosc

    D = -np.dot(normalny, p1)

    return normalny, D


# ==================================================
# 4. ODLEGLOSCI PUNKTOW OD PLASZCZYZNY
# ==================================================

def odleglosci_od_plaszczyzny(punkty, normalny, D):

    return np.abs(
        np.dot(punkty, normalny) + D
    )


# ==================================================
# 5. WLASNA IMPLEMENTACJA RANSAC
# ==================================================

def ransac_wlasny(punkty, iteracje=1000, prog=0.05):

    if len(punkty) < 3:
        return None

    najlepsze_inliery = None
    maks_inlierow = 0

    rng = np.random.default_rng(42)

    for i in range(iteracje):

        indeksy = rng.choice(
            len(punkty),
            3,
            replace=False
        )

        p1, p2, p3 = punkty[indeksy]

        wynik = oblicz_plaszczyzne(p1, p2, p3)

        if wynik is None:
            continue

        normalny, D = wynik

        odleglosci = odleglosci_od_plaszczyzny(
            punkty,
            normalny,
            D
        )

        inliery = odleglosci < prog

        liczba_inlierow = np.sum(inliery)

        if liczba_inlierow > maks_inlierow:

            maks_inlierow = liczba_inlierow
            najlepsze_inliery = inliery

    if najlepsze_inliery is None or maks_inlierow < 3:
        return None

    # Ponowne dopasowanie plaszczyzny
    # na podstawie znalezionych inlierow

    punkty_inliery = punkty[najlepsze_inliery]

    srodek = np.mean(punkty_inliery, axis=0)

    _, _, vh = np.linalg.svd(
        punkty_inliery - srodek
    )

    normalny = vh[-1]

    D = -np.dot(normalny, srodek)

    return oblicz_statystyki(
        punkty,
        normalny,
        D,
        prog
    )


# ==================================================
# 6. RANSAC Z BIBLIOTEKI PYRANSAC3D
# ==================================================

def ransac_biblioteka(punkty, iteracje=1000, prog=0.05):

    if len(punkty) < 3:
        return None

    plane = pyrsc.Plane()

    rownanie, inliery = plane.fit(
        punkty,
        thresh=prog,
        maxIteration=iteracje
    )

    if rownanie is None or len(inliery) < 3:
        return None

    A, B, C, D = rownanie

    normalny = np.array([A, B, C], dtype=float)

    dlugosc = np.linalg.norm(normalny)

    if dlugosc < 1e-10:
        return None

    normalny = normalny / dlugosc
    D = D / dlugosc

    return oblicz_statystyki(
        punkty,
        normalny,
        D,
        prog
    )


# ==================================================
# 7. OBLICZANIE STATYSTYK PLASZCZYZNY
# ==================================================

def oblicz_statystyki(punkty, normalny, D, prog):

    odleglosci = odleglosci_od_plaszczyzny(
        punkty,
        normalny,
        D
    )

    srednia = np.mean(odleglosci)

    udzial = np.mean(odleglosci < prog)

    liczba_inlierow = np.sum(odleglosci < prog)

    return {
        "normalny": normalny,
        "D": D,
        "srednia": srednia,
        "udzial": udzial,
        "inliery": liczba_inlierow
    }


# ==================================================
# 8. KLASYFIKACJA POWIERZCHNI
# ==================================================

def klasyfikuj_powierzchnie(wynik):

    normalny = wynik["normalny"]
    srednia = wynik["srednia"]
    udzial = wynik["udzial"]

    if srednia > 0.05 or udzial < 0.90:
        return "Nie jest plaszczyzna"

    nz = normalny[2]

    if abs(nz) > 0.95:
        return "Plaszczyzna pozioma"

    elif abs(nz) < 0.10:
        return "Plaszczyzna pionowa"

    else:
        return "Plaszczyzna pochyla"


# ==================================================
# 9. SEGMENTACJA K-MEANS
# ==================================================

def segmentacja_kmeans(punkty):

    print("\nRozpoczynam segmentacje K-means...")

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    etykiety = kmeans.fit_predict(punkty)

    return etykiety


# ==================================================
# 10. SEGMENTACJA DBSCAN
# ==================================================

def segmentacja_dbscan(punkty):

    print("\nRozpoczynam segmentacje DBSCAN...")

    dbscan = DBSCAN(
        eps=EPS_DBSCAN,
        min_samples=MIN_SAMPLES_DBSCAN
    )

    etykiety = dbscan.fit_predict(punkty)

    return etykiety


# ==================================================
# 11. ANALIZA CHMUR
# ==================================================

def analizuj_chmury(punkty, etykiety, metoda_ransac):

    wyniki = []

    numery_chmur = sorted(
        set(etykiety) - {-1}
    )

    for numer in numery_chmur:

        grupa = punkty[etykiety == numer]

        print("\n------------------------------")
        print("CHMURA", numer + 1)
        print("------------------------------")

        print("Liczba punktow:", len(grupa))

        if metoda_ransac == "wlasny":

            wynik = ransac_wlasny(
                grupa,
                iteracje=LICZBA_ITERACJI,
                prog=PROG_RANSAC
            )

        else:

            wynik = ransac_biblioteka(
                grupa,
                iteracje=LICZBA_ITERACJI,
                prog=PROG_RANSAC
            )

        if wynik is None:

            print("Nie znaleziono plaszczyzny")
            continue

        klasyfikacja = klasyfikuj_powierzchnie(wynik)

        print("Wektor normalny:", wynik["normalny"])
        print("Wspolczynnik D:", wynik["D"])
        print("Srednia odleglosc:", wynik["srednia"])
        print("Udzial inlierow:", wynik["udzial"])
        print("Liczba inlierow:", wynik["inliery"])
        print("Klasyfikacja:", klasyfikacja)

        wyniki.append({
            "numer": numer + 1,
            "liczba_punktow": len(grupa),
            "klasyfikacja": klasyfikacja,
            "srednia": wynik["srednia"],
            "udzial": wynik["udzial"],
            "normalny": wynik["normalny"]
        })

    return wyniki


# ==================================================
# 12. WIZUALIZACJA POROWNAWCZA
# ==================================================

def wizualizuj_porownanie(punkty, etykiety_kmeans, etykiety_dbscan):

    fig = plt.figure(figsize=(16, 7))

    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    kolory = plt.get_cmap("tab10")

    # K-MEANS
    for i in sorted(set(etykiety_kmeans)):

        grupa = punkty[etykiety_kmeans == i]

        ax1.scatter(
            grupa[:, 0],
            grupa[:, 1],
            grupa[:, 2],
            s=3,
            color=kolory(i % 10),
            label=f"Chmura {i + 1}"
        )

    ax1.set_title("METODA 1: K-means + wlasny RANSAC")

    # DBSCAN
    for i in sorted(set(etykiety_dbscan)):

        grupa = punkty[etykiety_dbscan == i]

        if i == -1:
            kolor = "black"
            nazwa = "Szum"

        else:
            kolor = kolory(i % 10)
            nazwa = f"Chmura {i + 1}"

        ax2.scatter(
            grupa[:, 0],
            grupa[:, 1],
            grupa[:, 2],
            s=3,
            color=kolor,
            label=nazwa
        )

    ax2.set_title("METODA 2: DBSCAN + pyransac3d")

    zakresy = np.maximum(np.ptp(punkty, axis=0), 0.01)

    for ax in [ax1, ax2]:

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_box_aspect(zakresy)

        ax.legend()

    plt.tight_layout()
    plt.show()


# ==================================================
# 13. POROWNANIE WYNIKOW
# ==================================================

def porownaj_wyniki(wyniki1, wyniki2, czas1, czas2,
                   etykiety1, etykiety2):

    print("\n")
    print("=" * 65)
    print("                 POROWNANIE METOD")
    print("=" * 65)

    liczba1 = len(set(etykiety1) - {-1})
    liczba2 = len(set(etykiety2) - {-1})

    szum1 = np.sum(etykiety1 == -1)
    szum2 = np.sum(etykiety2 == -1)

    print(f"{'PARAMETR':<25} {'METODA 1':<18} {'METODA 2':<18}")
    print("-" * 65)

    print(f"{'Segmentacja':<25} {'K-means':<18} {'DBSCAN':<18}")
    print(f"{'RANSAC':<25} {'Wlasny':<18} {'pyransac3d':<18}")

    print(f"{'Liczba chmur':<25} {liczba1:<18} {liczba2:<18}")
    print(f"{'Punkty szumu':<25} {szum1:<18} {szum2:<18}")

    print(f"{'Czas [s]':<25} {czas1:<18.4f} {czas2:<18.4f}")

    print("-" * 65)

    print("\nKLASYFIKACJA POWIERZCHNI:")

    print("\nMETODA 1:")
    for wynik in wyniki1:
        print(
            f"Chmura {wynik['numer']}: "
            f"{wynik['klasyfikacja']}"
        )

    print("\nMETODA 2:")
    for wynik in wyniki2:
        print(
            f"Chmura {wynik['numer']}: "
            f"{wynik['klasyfikacja']}"
        )

    print("\nUWAGA: numery skupisk miedzy metodami")
    print("nie musza odpowiadac tym samym obiektom.")


# ==================================================
# 14. PROGRAM GLOWNY
# ==================================================

if __name__ == "__main__":

    print("=" * 65)
    print("POROWNANIE K-MEANS + RANSAC ORAZ DBSCAN + PYRANSAC3D")
    print("=" * 65)

    # Wczytanie danych
    punkty = polacz_chmury()

    # ==============================================
    # METODA 1
    # K-MEANS + WLASNY RANSAC
    # ==============================================

    print("\n")
    print("=" * 65)
    print("METODA 1 - K-MEANS + WLASNY RANSAC")
    print("=" * 65)

    start1 = time.perf_counter()

    etykiety_kmeans = segmentacja_kmeans(punkty)

    wyniki1 = analizuj_chmury(
        punkty,
        etykiety_kmeans,
        "wlasny"
    )

    czas1 = time.perf_counter() - start1

    print("\nCzas metody 1:", round(czas1, 4), "s")

    # ==============================================
    # METODA 2
    # DBSCAN + PYRANSAC3D
    # ==============================================

    print("\n")
    print("=" * 65)
    print("METODA 2 - DBSCAN + PYRANSAC3D")
    print("=" * 65)

    start2 = time.perf_counter()

    etykiety_dbscan = segmentacja_dbscan(punkty)

    wyniki2 = analizuj_chmury(
        punkty,
        etykiety_dbscan,
        "biblioteka"
    )

    czas2 = time.perf_counter() - start2

    print("\nCzas metody 2:", round(czas2, 4), "s")

    # ==============================================
    # POROWNANIE WYNIKOW
    # ==============================================

    porownaj_wyniki(
        wyniki1,
        wyniki2,
        czas1,
        czas2,
        etykiety_kmeans,
        etykiety_dbscan
    )

    # ==============================================
    # WIZUALIZACJA
    # ==============================================

    wizualizuj_porownanie(
        punkty,
        etykiety_kmeans,
        etykiety_dbscan
    )