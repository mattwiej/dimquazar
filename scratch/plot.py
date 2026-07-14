import pandas as pd
import matplotlib.pyplot as plt

# Lista plików do wczytania i kolory dla każdego modelu
files = ['testModel.txt', 'best_testModel.txt', 'max_testModel.txt', 'min_testModel.txt']
colors = ['blue', 'green', 'red', 'orange']

# Ustawienie rozmiaru wykresu
plt.figure(figsize=(12, 8))

# Pętla przechodząca przez każdy plik
for file, color in zip(files, colors):
    try:
        # Wczytanie danych (zakładamy, że kolumny oddzielone są spacjami, a komentarze zaczynają się od '#')
        df = pd.read_csv(file, sep='\s+', comment='#', header=None)
        
        # Narysowanie danych (kolumna 0 to X, kolumna 1 to Y)
        plt.plot(df[0], df[1], label=file.replace('.txt', ''), color=color, alpha=0.7)
    except Exception as e:
        print(f"Błąd podczas czytania pliku {file}: {e}")

# Dodanie etykiet osi, tytułu, legendy i siatki
plt.xlabel('Długość fali ($\AA$)')
plt.ylabel('Gęstość strumienia ($10^{-16}$ erg/s/cm$^2$/$\AA$)')
plt.title('Porównanie gęstości strumienia dla różnych modeli')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Zapisanie wykresu do pliku (możesz użyć plt.show(), aby wyświetlić w oknie)
plt.savefig('combined_plot.png')
plt.show() # Odkomentuj to, jeśli chcesz wyświetlić wykres na ekranie zamiast tylko go zapisywać
