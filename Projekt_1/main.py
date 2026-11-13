import numpy as np
import matplotlib.pyplot as plt


# A. Powierzchnia pozioma
def powierzchnia_pozioma(szerokosc, dlugosc, liczba_punktow=1000):
    x = np.random.uniform(0, szerokosc, liczba_punktow)
    y = np.random.uniform(0, dlugosc, liczba_punktow)
    z = np.zeros(liczba_punktow)

    return np.column_stack((x, y, z))


# B. Powierzchnia pionowa
def powierzchnia_pionowa(szerokosc, wysokosc, liczba_punktow=1000):
    x = np.random.uniform(0, szerokosc, liczba_punktow)
    y = np.zeros(liczba_punktow)
    z = np.random.uniform(0, wysokosc, liczba_punktow)

    return np.column_stack((x, y, z))


# C. Powierzchnia cylindryczna
def powierzchnia_cylindryczna(promien, wysokosc, liczba_punktow=1000):
    kat = np.random.uniform(0, 2 * np.pi, liczba_punktow)

    x = promien * np.cos(kat)
    y = promien * np.sin(kat)
    z = np.random.uniform(0, wysokosc, liczba_punktow)

    return np.column_stack((x, y, z))


# Zapisywanie chmury punktów do pliku XYZ
def zapisz_xyz(nazwa_pliku, punkty):
    np.savetxt(nazwa_pliku, punkty, fmt="%.6f")


# Wizualizacja chmury punktów
def pokaz_chmure(punkty, tytul):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        punkty[:, 0],
        punkty[:, 1],
        punkty[:, 2],
        s=2
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(tytul)
    ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":

    # Generowanie punktów
    pozioma = powierzchnia_pozioma(10, 5, 1000)
    pionowa = powierzchnia_pionowa(10, 5, 1000)
    cylinder = powierzchnia_cylindryczna(3, 10, 1000)

    # Zapisywanie plików XYZ
    zapisz_xyz("pozioma.xyz", pozioma)
    zapisz_xyz("pionowa.xyz", pionowa)
    zapisz_xyz("cylinder.xyz", cylinder)

    print("Zapisano chmury punktów!")

    # Wizualizacja
    pokaz_chmure(pozioma, "Powierzchnia pozioma")
    pokaz_chmure(pionowa, "Powierzchnia pionowa")
    pokaz_chmure(cylinder, "Powierzchnia cylindryczna")