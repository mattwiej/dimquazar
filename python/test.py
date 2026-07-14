import math
import random
import os
import time

class Result:
    __slots__ = ['chi2', 'amplitude', 'sigma', 'x0', 'alpha']
    def __init__(self, chi2=float('inf'), amplitude=0.0, sigma=0.0, x0=0.0, alpha=0.0):
        self.chi2 = chi2
        self.amplitude = amplitude
        self.sigma = sigma
        self.x0 = x0
        self.alpha = alpha

class FinalResult(Result):
    __slots__ = ['minAlpha', 'maxAlpha', 'minAmplitude', 'maxAmplitude', 'minSigma', 'maxSigma', 'minX0', 'maxX0']
    def __init__(self):
        super().__init__()
        self.minAlpha = float('inf')
        self.maxAlpha = float('-inf')
        self.minAmplitude = float('inf')
        self.maxAmplitude = float('-inf')
        self.minSigma = float('inf')
        self.maxSigma = float('-inf')
        self.minX0 = float('inf')
        self.maxX0 = float('-inf')

class Config:
    def __init__(self):
        self.dataPath = ""
        self.resultsDirPath = ""
        self.modelsDirPath = ""
        self.iterationCount = 0
        self.percentOfBestChi2 = 0.0
        self.parametrCount = 0.0
        self.chiThreshold = 0.0
        self.seed = 0
        
        # === Gauss
        self.windowX = 0.0
        self.windowWidth = 0.0
        self.amplitudeGuess = 0.0
        self.amplitudeWidth = 0.0
        self.sigmaGuess = 0.0
        self.sigmaWidth = 0.0
        self.x0Guess = 0.0
        self.x0Width = 0.0
        
        # === Cont
        self.contLambda0 = 0.0
        self.contB = 0.0
        self.contAlphaGuess = 0.0
        self.contAlphaWidth = 0.0
        
        # === Results
        self.numberOfPoints = 0.0
        self.finalResult = FinalResult()
        self.results = []

class Data:
    def __init__(self):
        self.lambda_arr = []
        self.flux = []

def chi_sqr(experiment, expected):
    temp = experiment - expected
    return (temp * temp) / expected

def keep_best_results(arr, t):
    # Odrzucenie jeśli chi2 jest gorsze niż najgorszy zapisany element
    if t.chi2 >= arr[-1].chi2:
        return

    # Binary search dla optymalnego miejsca wstawienia (wyniki posortowane rosnąco po chi2)
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (high + low) // 2
        if arr[mid].chi2 < t.chi2:
            low = mid + 1
        else:
            high = mid - 1

    insert_idx = low

    if insert_idx < len(arr):
        # Wstaw element i usuń ostatni, aby zachować stały rozmiar tablicy
        arr.insert(insert_idx, t)
        arr.pop()

def set_seed_of_the_prng(c, seed=0):
    if seed == 0:
        c.seed = int(time.time() * 1_000_000_000) # nanosekundy
    else:
        c.seed = seed
    random.seed(c.seed)

def find_min_max_parameters(arr, c):
    minAlpha = float('inf')
    maxAlpha = float('-inf')
    minAmplitude = float('inf')
    maxAmplitude = float('-inf')
    minSigma = float('inf')
    maxSigma = float('-inf')
    minX0 = float('inf')
    maxX0 = float('-inf')

    for e in arr:
        if e.chi2 == float('inf'): # Pomijamy niewypełnione wyniki
            continue
            
        if e.alpha < minAlpha: minAlpha = e.alpha
        if e.alpha > maxAlpha: maxAlpha = e.alpha

        if e.amplitude < minAmplitude: minAmplitude = e.amplitude
        if e.amplitude > maxAmplitude: maxAmplitude = e.amplitude

        if e.sigma < minSigma: minSigma = e.sigma
        if e.sigma > maxSigma: maxSigma = e.sigma

        if e.x0 < minX0: minX0 = e.x0
        if e.x0 > maxX0: maxX0 = e.x0

    c.finalResult.minAlpha = minAlpha
    c.finalResult.maxAlpha = maxAlpha
    c.finalResult.minAmplitude = minAmplitude
    c.finalResult.maxAmplitude = maxAmplitude
    c.finalResult.minSigma = minSigma
    c.finalResult.maxSigma = maxSigma
    c.finalResult.minX0 = minX0
    c.finalResult.maxX0 = maxX0

def monte_carlo(c, dane):
    best = Result()
    # oneOverSqrt2Pi = 1 / math.sqrt(2 * math.pi) # Pozostawione jako komentarz, tak jak w Odinie
    start_idx = -1
    end_idx = -1

    print("Liczenie punktów...")
    for j in range(len(dane.lambda_arr)):
        lam = dane.lambda_arr[j]
        if c.windowX - c.windowWidth <= lam <= c.windowX + c.windowWidth:
            if start_idx == -1:
                start_idx = j
            c.numberOfPoints += 1
            end_idx = j

    if start_idx == -1:
        return
        
    points_in_window = end_idx - start_idx + 1

    precalc_ln = [0.0] * points_in_window
    for j in range(points_in_window):
        lam = dane.lambda_arr[start_idx + j]
        precalc_ln[j] = math.log(lam / c.contLambda0)

    print("Uruchamianie pętli Monte Carlo...")
    start_time = time.perf_counter()

    for i in range(c.iterationCount):
        temp = Result(chi2=0.0)
        
        # Losowanie parametrów z clamp
        clamp_val = max(0, min(c.amplitudeGuess, c.amplitudeGuess - c.amplitudeWidth))
        temp.amplitude = random.uniform(clamp_val, c.amplitudeGuess + c.amplitudeWidth)
        temp.x0 = random.uniform(c.x0Guess - c.x0Width, c.x0Guess + c.x0Width)
        temp.sigma = random.uniform(c.sigmaGuess - c.sigmaWidth, c.sigmaGuess + c.sigmaWidth)
        temp.alpha = random.uniform(c.contAlphaGuess - c.contAlphaWidth, c.contAlphaGuess + c.contAlphaWidth)

        # A = temp.amplitude / temp.sigma * oneOverSqrt2Pi
        twoSigmaSqr = 2 * temp.sigma * temp.sigma

        for j in range(points_in_window):
            lam = dane.lambda_arr[start_idx + j]
            
            # Kontinuum
            cont = c.contB * math.exp(-temp.alpha * precalc_ln[j])
            tempFlux = dane.flux[start_idx + j] - cont

            # Zgodnie z oryginałem w Odinie - użyto temp.amplitude bezpośrednio
            lambdaDiff = lam - temp.x0
            exponent = -(lambdaDiff * lambdaDiff) / twoSigmaSqr
            modelFlux = temp.amplitude * math.exp(exponent)
            
            temp.chi2 += chi_sqr(tempFlux, modelFlux)

        # Sprawdzanie i zapisywanie wyników
        if (temp.chi2 / (c.numberOfPoints - c.parametrCount) < c.chiThreshold):
            keep_best_results(c.results, temp)

        if temp.chi2 < best.chi2:
            # W Pythonie kopiowanie obiektów wymaga przepisania lub użycia copy
            best = Result(temp.chi2, temp.amplitude, temp.sigma, temp.x0, temp.alpha)

    end_time = time.perf_counter()
    print(f"Pętla Monte Carlo zajęła: {end_time - start_time:.4f} s")

    # Przekazanie wartości best do finalResult
    c.finalResult.chi2 = best.chi2
    c.finalResult.amplitude = best.amplitude
    c.finalResult.sigma = best.sigma
    c.finalResult.x0 = best.x0
    c.finalResult.alpha = best.alpha

    print("Szukanie Min/Max...")
    find_min_max_parameters(c.results, c)

def load_data_from_file(path, dane):
    try:
        with open(path, 'r') as file:
            for line in file:
                trimmed = line.strip()
                if len(trimmed) == 0 or trimmed.startswith('#'):
                    continue
                columns = trimmed.split()
                if len(columns) == 2:
                    try:
                        lam = float(columns[0])
                        flux = float(columns[1])
                        dane.lambda_arr.append(lam)
                        dane.flux.append(flux)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"Failed to read the file: {path}")

def save_results_to_file(c, label="chi2 amplitude sigma x0 alpha"):
    os.makedirs(c.resultsDirPath, exist_ok=True)
    filename = os.path.basename(c.dataPath)
    new_name = f"wynik_{filename}"
    path = os.path.join(c.resultsDirPath, new_name)

    try:
        with open(path, 'w') as file:
            file.write(f"# seed: {c.seed}\n")
            file.write(f"# N: {c.numberOfPoints}\n")
            file.write(f"# minAmplitude: {c.finalResult.minAmplitude}\n")
            file.write(f"# maxAmplitude: {c.finalResult.maxAmplitude}\n")
            file.write(f"# minSigma: {c.finalResult.minSigma}\n")
            file.write(f"# maxSigma: {c.finalResult.maxSigma}\n")
            file.write(f"# minX0: {c.finalResult.minX0}\n")
            file.write(f"# maxX0: {c.finalResult.maxX0}\n")
            file.write(f"# minAlpha: {c.finalResult.minAlpha}\n")
            file.write(f"# maxAlpha: {c.finalResult.maxAlpha}\n")
            file.write(f"# {label}\n")

            for res in c.results:
                if res.chi2 != float('inf'): # Piszemy tylko zapisane wyniki
                    file.write(f"{res.chi2:.7f} {res.amplitude:.7f} {res.sigma:.7f} {res.x0:.7f} {res.alpha:.7f}\n")
        print(f"Saved to file: {path}")
    except Exception as e:
        print(f"Failed to save to file: {path} - {e}")

def add_gauss(sigma, amplitude, x0, lambda_arr, flux_arr, start, end):
    twoSigSqr = 2 * sigma * sigma
    for i in range(start, end + 1):
        lambdaDiff = lambda_arr[i] - x0
        exponent = -(lambdaDiff * lambdaDiff) / twoSigSqr
        flux_arr[i] += amplitude * math.exp(exponent)

def save_model_to_file(lambda_arr, flux_arr, label, c):
    os.makedirs(c.modelsDirPath, exist_ok=True)
    filename = os.path.basename(c.dataPath)
    new_name = f"{label}_{filename}"
    path = os.path.join(c.modelsDirPath, new_name)

    try:
        with open(path, 'w') as file:
            file.write("# Wavelength(A) Flux_density(10^{-16} erg/s^{-1}/cm^{-2}/A^{-1})\n")
            for i in range(len(lambda_arr)):
                # Format %e wymusza notację naukową tak jak w Odinie
                file.write(f"{lambda_arr[i]:.7f} {flux_arr[i]:e}\n")
        print(f"Saved to file: {path}")
    except Exception as e:
        print(f"Failed to save to file: {path} - {e}")

def generate_mathematical_models(dane, c):
    modelFlux = [0.0] * len(dane.lambda_arr)
    start_idx = -1
    end_idx = -1

    bestSigma = c.finalResult.sigma
    bestAmplitude = c.finalResult.amplitude
    bestAlpha = c.finalResult.alpha
    bestX0 = c.finalResult.x0

    minSigma = c.finalResult.minSigma
    minAmplitude = c.finalResult.minAmplitude
    maxSigma = c.finalResult.maxSigma
    maxAmplitude = c.finalResult.maxAmplitude

    for i in range(len(dane.lambda_arr)):
        lam = dane.lambda_arr[i]
        modelFlux[i] = c.contB * math.pow((lam / c.contLambda0), -bestAlpha)
        if c.windowX - c.windowWidth <= lam <= c.windowX + c.windowWidth:
            if start_idx == -1:
                start_idx = i
            end_idx = i

    minFlux = list(modelFlux)
    maxFlux = list(modelFlux)

    add_gauss(bestSigma, bestAmplitude, bestX0, dane.lambda_arr, modelFlux, start_idx, end_idx)
    add_gauss(minSigma, minAmplitude, bestX0, dane.lambda_arr, minFlux, start_idx, end_idx)
    add_gauss(maxSigma, maxAmplitude, bestX0, dane.lambda_arr, maxFlux, start_idx, end_idx)

    save_model_to_file(dane.lambda_arr, modelFlux, "best", c)
    save_model_to_file(dane.lambda_arr, minFlux, "min", c)
    save_model_to_file(dane.lambda_arr, maxFlux, "max", c)

def main():
    print("Inicjalizacja konfiguracji...")
    konfiguracja = Config()

    konfiguracja.windowX = 2801
    konfiguracja.windowWidth = 50
    konfiguracja.chiThreshold = 3
    konfiguracja.parametrCount = 4

    konfiguracja.contLambda0 = 3000
    konfiguracja.contB = 2.128
    konfiguracja.contAlphaGuess = 1.556
    konfiguracja.contAlphaWidth = 0.1

    konfiguracja.amplitudeGuess = 1
    konfiguracja.amplitudeWidth = 3
    konfiguracja.x0Guess = 2800
    konfiguracja.x0Width = 50
    konfiguracja.sigmaGuess = 16
    konfiguracja.sigmaWidth = 2

    konfiguracja.iterationCount = 1_000_000
    konfiguracja.percentOfBestChi2 = 0.01
    
    length = int(konfiguracja.iterationCount * konfiguracja.percentOfBestChi2)
    konfiguracja.results = [Result() for _ in range(length)]

    dane = Data()

    konfiguracja.resultsDirPath = "./wyniki/"
    konfiguracja.modelsDirPath = "./modele/"
    konfiguracja.dataPath = "./dane/testModel.txt"

    print("Seedowanie RNG...")
    set_seed_of_the_prng(konfiguracja)

    print("Ładowanie danych z pliku...")
    load_data_from_file(konfiguracja.dataPath, dane)
    
    if len(dane.lambda_arr) == 0:
        print("Brak danych do analizy. Upewnij się, że ścieżka do pliku jest poprawna.")
        return

    monte_carlo(konfiguracja, dane)

    print(f"Best Chi2: {konfiguracja.finalResult.chi2}")
    
    print("Zapisywanie...")
    save_results_to_file(konfiguracja)
    
    print("Modelowanie Matematyczne...")
    generate_mathematical_models(dane, konfiguracja)

if __name__ == "__main__":
    total_start = time.perf_counter()
    main()
    print(f"Całkowity czas wykonania programu: {time.perf_counter() - total_start:.4f} s")
