import numpy as np
import os
import time

def main():
    print("Inicjalizacja środowiska...")
    
    # === KONFIGURACJA (Płaska struktura, klasyka skryptów badawczych) ===
    # Gauss
    window_x = 2801.0
    window_width = 50.0
    amp_guess = 1.0
    amp_width = 3.0
    sigma_guess = 16.0
    sigma_width = 2.0
    x0_guess = 2800.0
    x0_width = 50.0
    
    # Continuum
    cont_lambda0 = 3000.0
    cont_b = 2.128
    cont_alpha_guess = 1.556
    cont_alpha_width = 0.1
    
    # Monte Carlo parametry
    iterations = 1_000_000
    percent_of_best = 0.01
    chi_threshold = 3.0
    param_count = 4
    
    data_path = "./dane/testModel.txt"
    results_dir = "./wyniki/"
    models_dir = "./modele/"
    
    # Ustawienie ziarna generatora (NumPy ma swój własny, szybki moduł random)
    np.random.seed(int(time.time()))

    # === WCZYTYWANIE DANYCH ===
    print("Ładowanie danych wejściowych...")
    try:
        # np.loadtxt rozwiązuje 99% problemów z plikami txt
        data = np.loadtxt(data_path, comments='#')
    except OSError:
        print(f"Brak pliku: {data_path}!")
        return
        
    lam_full = data[:, 0]
    flux_full = data[:, 1]

    # Zamiast szukać indeksów ręcznie, tworzymy "maskę logiczną" dla interesującego nas okna
    mask = (lam_full >= window_x - window_width) & (lam_full <= window_x + window_width)
    lam_win = lam_full[mask]
    flux_win = flux_full[mask]
    N = len(lam_win)

    if N == 0:
        print("Brak punktów w zadanym oknie lambdy.")
        return

    # Wcześniejsze wyliczenie logarytmów (wektorowo!)
    precalc_ln = np.log(lam_win / cont_lambda0)

    # === MONTE CARLO ===
    print(f"Uruchamianie pętli Monte Carlo (Iteracje: {iterations})...")
    start_time = time.perf_counter()

    results_list = []
    best_chi2 = np.inf
    best_params = None

    # Mimo że pętla w Pythonie jest wolna, wewnątrz mamy wektoryzację (operacje na macierzach).
    for i in range(iterations):
        # Losowanie parametrów z użyciem funkcji uniform
        amp = np.random.uniform(max(0, amp_guess - amp_width), amp_guess + amp_width)
        x0 = np.random.uniform(x0_guess - x0_width, x0_guess + x0_width)
        sigma = np.random.uniform(sigma_guess - sigma_width, sigma_guess + sigma_width)
        alpha = np.random.uniform(cont_alpha_guess - cont_alpha_width, cont_alpha_guess + cont_alpha_width)

        two_sigma_sqr = 2 * sigma**2

        # 1. Liczymy continuum wektorowo dla wszystkich punktów naraz
        cont = cont_b * np.exp(-alpha * precalc_ln)
        temp_flux = flux_win - cont

        # 2. Liczymy model strumienia z rozkładem Gaussa (wektorowo)
        model_flux = amp * np.exp(-((lam_win - x0)**2) / two_sigma_sqr)

        # 3. Liczymy Chi-kwadrat (sumowanie całej tablicy naraz)
        chi2 = np.sum(((temp_flux - model_flux)**2) / model_flux)

        # Zbieramy wyniki poniżej progu
        if (chi2 / (N - param_count)) < chi_threshold:
            results_list.append([chi2, amp, sigma, x0, alpha])

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_params = [chi2, amp, sigma, x0, alpha]

    end_time = time.perf_counter()
    print(f"Pętla Monte Carlo zakończona w: {end_time - start_time:.4f} s")

    # === OBRÓBKA WYNIKÓW I ZAPIS ===
    if not results_list:
        print("Żaden z losowych modeli nie spełnił progu chiThreshold.")
        return

    # Zamieniamy listę na wydajną macierz 2D
    res_arr = np.array(results_list)
    
    # Sortujemy po pierwszej kolumnie (chi2) i obcinamy do najlepszego procenta
    res_arr = res_arr[res_arr[:, 0].argsort()]
    limit = max(1, int(iterations * percent_of_best))
    top_results = res_arr[:limit]

    print("Znajdywanie wartości minimalnych i maksymalnych...")
    # np.min i np.max działają na całych kolumnach na raz (axis=0)
    min_vals = np.min(top_results, axis=0)
    max_vals = np.max(top_results, axis=0)

    # Przygotowanie katalogów
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Zapis wyników do pliku...")
    filename = os.path.basename(data_path)
    
    # Zbudowanie zgrabnego nagłówka
    header = (f"seed: (numpy internal)\nN: {N}\n"
              f"minAmplitude: {min_vals[1]:.7f}\nmaxAmplitude: {max_vals[1]:.7f}\n"
              f"minSigma: {min_vals[2]:.7f}\nmaxSigma: {max_vals[2]:.7f}\n"
              f"minX0: {min_vals[3]:.7f}\nmaxX0: {max_vals[3]:.7f}\n"
              f"minAlpha: {min_vals[4]:.7f}\nmaxAlpha: {max_vals[4]:.7f}\n"
              "chi2 amplitude sigma x0 alpha")
    
    np.savetxt(os.path.join(results_dir, f"wynik_{filename}"), top_results, fmt="%.7f", header=header)

    # === GENEROWANIE MATEMATYCZNYCH MODELI Z NAKŁADKĄ GAUSSA ===
    print("Generowanie i zapis modeli matematycznych...")
    best_amp, best_sig, best_x0, best_alpha = best_params[1:]
    
    # Funkcja pomocnicza generująca pełny wektor (ułatwia sprawę)
    def generate_full_model(alpha, amp, sigma, x0):
        # Continuum dla całego zakresu
        base_model = cont_b * (lam_full / cont_lambda0)**(-alpha)
        # Maska Gaussa tylko dla odpowiednich punktów
        gauss_part = np.zeros_like(lam_full)
        gauss_part[mask] = amp * np.exp(-((lam_win - x0)**2) / (2 * sigma**2))
        return base_model + gauss_part

    # Tworzenie krzywych
    flux_best = generate_full_model(best_alpha, best_amp, best_sig, best_x0)
    flux_min  = generate_full_model(best_alpha, min_vals[1], min_vals[2], best_x0)
    flux_max  = generate_full_model(best_alpha, max_vals[1], max_vals[2], best_x0)

    # Zapis
    model_header = "Wavelength(A) Flux_density(10^{-16} erg/s^{-1}/cm^{-2}/A^{-1})"
    np.savetxt(os.path.join(models_dir, f"best_{filename}"), np.column_stack((lam_full, flux_best)), fmt="%.7f %e", header=model_header)
    np.savetxt(os.path.join(models_dir, f"min_{filename}"), np.column_stack((lam_full, flux_min)), fmt="%.7f %e", header=model_header)
    np.savetxt(os.path.join(models_dir, f"max_{filename}"), np.column_stack((lam_full, flux_max)), fmt="%.7f %e", header=model_header)

    print("Koniec obliczeń!")

if __name__ == "__main__":
    main()
