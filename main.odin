package main

import "core:encoding/json"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strconv"
import "core:strings"
import "core:time"

data :: struct {
	lambda: [dynamic]f64,
	flux:   [dynamic]f64,
}
config :: struct {
	dataPath:          string,
	resultsDirPath:    string,
	modelsDirPath:     string,
	iterationCount:    i32,
	percentOfBestChi2: f64,
	parametrCount:     f64,
	chiThreshold:      f64,
	seed:              u64,
	//===Gauss
	windowX:           f64,
	windowWidth:       f64, // NOTE: +/- windowX
	amplitudeGuess:    f64,
	amplitudeWidth:    f64,
	sigmaGuess:        f64,
	sigmaWidth:        f64,
	x0Guess:           f64,
	x0Width:           f64,
	//===Cont
	contLambda0:       f64,
	contB:             f64,
	contAlphaGuess:    f64,
	contAlphaWidth:    f64,
	//===Results
	numberOfPoints:    f64,
	finalResult:       finalResult,
	results:           [dynamic]result,
}

finalResult :: struct {
	using result: result,
	minAlpha:     f64,
	maxAlpha:     f64,
	minAmplitude: f64,
	maxAmplitude: f64,
	minSigma:     f64,
	maxSigma:     f64,
	minX0:        f64,
	maxX0:        f64,
}

result :: struct {
	chi2:      f64,
	amplitude: f64,
	sigma:     f64,
	x0:        f64,
	alpha:     f64,
}

main :: proc() {

	prof_init()
	defer prof_deinit()

	prof_begin("Program")
	konfiguracja: config

	konfiguracja.windowX = 2801
	konfiguracja.windowWidth = 50
	konfiguracja.chiThreshold = 3
	konfiguracja.parametrCount = 4

	konfiguracja.contLambda0 = 3000
	konfiguracja.contB = 2.128 // NOTE: Flux musi byc w jednostkch 10^-16
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
	length := f64(konfiguracja.iterationCount) * konfiguracja.percentOfBestChi2 //length to dlugosc tablicy do konfiguracja.results

	konfiguracja.results = make([dynamic]result, i32(length), i32(length))

	prof_begin("zerowanie tablicy wynikow")
	for &w in konfiguracja.results {
		// NOTE: musimy dodac tu max f64 bo inaczej nie zapiszemy wynikow do tej tablicy,
		// powiewaz domyslnie pamiec w Odnie jest zerowana a chi^2 nie bedzie mniejsze od zera
		w.chi2 = math.F64_MAX
	}
	prof_end()
	dane: data
	dane.flux = make([dynamic]f64)
	dane.lambda = make([dynamic]f64)

	konfiguracja.resultsDirPath = "./wyniki/"
	konfiguracja.modelsDirPath = "./modele/"
	konfiguracja.dataPath = "./dane/testModel.txt"
	//konfiguracja.dataPath = "./dane/Block10.txt"
	//konfiguracja.dataPath = "./dane/Block5.txt"
	//konfiguracja.dataPath = "./dane/Moving10.txt"
	//konfiguracja.dataPath = "./dane/Moving5.txt"

	prof_begin("seedowanie rng")
	setSeedOfThePRNG(&konfiguracja)
	prof_end()

	prof_begin("ladowanie danych z pliku")
	loadDataFromFile(konfiguracja.dataPath, &dane)
	prof_end()

	prof_begin("montecarlo")
	monteCarlo(&konfiguracja, dane)
	prof_end()

	fmt.printf("Results: %#v\n", konfiguracja.finalResult)

	prof_begin("zapisywanie")
	saveResultsToFile(konfiguracja)
	prof_end()

	prof_begin("ModelowanieMatematyczne")
	generateMathematicalModels(dane, konfiguracja)
	prof_end()

	// NOTE: tak mozna na przyklad zrobic kilka losowan na raz

	//for i := 0; i < 9; i += 1 {
	//	for j in 0 ..< len(konfiguracja.results) {
	//		konfiguracja.results[j].chi2 = math.F64_MAX
	//	}
	//	konfiguracja.numberOfPoints = 0
	//	setSeedOfThePRNG(&konfiguracja)
	//	monteCarlo(&konfiguracja, dane)
	//	path := fmt.aprintf("wynikCSV%i.txt", i + 1)
	//	saveResultsToFile(konfiguracja, path)
	//}


	prof_end()
}

monteCarlo :: proc(c: ^config, dane: data) {

	best: result
	best.chi2 = math.F64_MAX
	oneOverSqrt2Pi: f64 = 1 / math.sqrt_f64(2 * math.PI)
	startIdx := -1
	endIdx := -1

	prof_begin("liczeniePkt")
	for j in 0 ..< len(dane.lambda) {
		lambda := dane.lambda[j]
		if lambda >= c.windowX - c.windowWidth && lambda <= c.windowX + c.windowWidth {
			if startIdx == -1 do startIdx = j
			c.numberOfPoints += 1
			endIdx = j
		}
	}
	prof_end()

	if startIdx == -1 do return
	pointsInWindow := endIdx - startIdx + 1


	precalc_ln := make([]f64, pointsInWindow, context.temp_allocator)
	for j in 0 ..< pointsInWindow {
		lambda := dane.lambda[startIdx + j]
		precalc_ln[j] = math.ln_f64(lambda / c.contLambda0)
	}


	for i in 0 ..< c.iterationCount {

		prof_begin("MonteCarlo Iter")
		//========Losowanie Parametrow
		temp: result
		temp.amplitude = rand.float64_uniform(
			math.clamp(c.amplitudeGuess - c.amplitudeWidth, 0, c.amplitudeGuess),
			c.amplitudeGuess + c.amplitudeWidth,
		)
		temp.x0 = rand.float64_uniform(c.x0Guess - c.x0Width, c.x0Guess + c.x0Width)
		temp.sigma = rand.float64_uniform(c.sigmaGuess - c.sigmaWidth, c.sigmaGuess + c.sigmaWidth)
		temp.alpha = rand.float64_uniform(
			c.contAlphaGuess - c.contAlphaWidth,
			c.contAlphaGuess + c.contAlphaWidth,
		)
		//========/Losowanie Parametrow

		//========Oblicznie parametrow ktore nie zmieniaja sie w ponizszej petli
		A := temp.amplitude / temp.sigma * oneOverSqrt2Pi
		twoSigmaSqr := 2 * temp.sigma * temp.sigma
		//========/Obliczanie

		// NOTE: Wczesniej obliczylismy indeksy ktore znajduja sie w WindowX +- WindowWidth
		// zatem teraz mozemy przejsc tylko po tych punktach

		#no_bounds_check for j in 0 ..< pointsInWindow {
			lambda := dane.lambda[startIdx + j]

			// NOTE: B * (lambda/lambda0)^-alpha jest wolniejsze niz
			// B exp(-alpha * ln(lambda/lambda0)) wiec korzystamy z tego
			cont := c.contB * math.exp_f64(-temp.alpha * precalc_ln[j])
			tempFlux := dane.flux[startIdx + j] - cont


			// NOTE: A/(sigma * sqrt(2*Pi) * exp[-(lambda-x0)^2/(2sigma^2)]
			lambdaDiff := lambda - temp.x0
			exponent := -(lambdaDiff * lambdaDiff) / twoSigmaSqr

			//modelFlux := A * math.exp_f64(exponent)
			modelFlux := temp.amplitude * math.exp_f64(exponent)
			temp.chi2 += chiSqr(tempFlux, modelFlux)

		}

		if (temp.chi2 / (c.numberOfPoints - c.parametrCount) < c.chiThreshold) {
			keepBestResults(c.results[:], temp)
		}

		if temp.chi2 < best.chi2 {
			best = temp
		}

		prof_end()
	}
	c.finalResult.result = best
	prof_begin("findMinMax")
	findMinMaxParameters(c.results[:], c)
	prof_end()
}


keepBestResults :: proc(arr: []result, t: result) {

	if t.chi2 >= arr[len(arr) - 1].chi2 do return

	//binary search
	low := 0
	high := len(arr) - 1

	for low <= high {
		mid := (high + low) / 2

		if arr[mid].chi2 < t.chi2 {
			low = mid + 1
		} else {
			high = mid - 1
		}
	}

	insertIdx := low

	//make room to insert, best chi2 at index 0
	if insertIdx < len(arr) {
		for i := len(arr) - 1; i > insertIdx; i -= 1 {
			arr[i] = arr[i - 1]
		}

		arr[insertIdx] = t
	}
}

setSeedOfThePRNG :: proc(c: ^config, seed: u64 = 0) {
	if seed == 0 {
		c.seed = u64(time.now()._nsec)
		rand.reset(c.seed)
		return
	}
	c.seed = seed
	rand.reset(c.seed)
}

//================== Stat
chiSqr :: proc(experiment, expected: f64) -> f64 {
	temp := (experiment - expected)
	chi := (temp * temp) / expected
	return chi
}

generateMathematicalModels :: proc(dane: data, c: config) {

	modelFlux := make([dynamic]f64, len(dane.lambda), len(dane.lambda))
	startIdx := -1
	endIdx := -1

	bestSigma := c.finalResult.sigma
	bestAmplitude := c.finalResult.amplitude
	bestAlpha := c.finalResult.alpha
	bestX0 := c.finalResult.x0

	minSigma := c.finalResult.minSigma
	minAmplitude := c.finalResult.minAmplitude

	maxSigma := c.finalResult.maxSigma
	maxAmplitude := c.finalResult.maxAmplitude

	for i := 0; i < len(dane.lambda); i += 1 {
		l := dane.lambda[i]
		modelFlux[i] = c.contB * math.pow((l / c.contLambda0), -bestAlpha)
		if l >= c.windowX - c.windowWidth && l <= c.windowX + c.windowWidth {
			if startIdx == -1 do startIdx = i
			endIdx = i
		}
	}

	minFlux := slice.clone(modelFlux[:])
	maxFlux := slice.clone(modelFlux[:])

	addGauss(bestSigma, bestAmplitude, bestX0, dane.lambda[:], modelFlux[:], startIdx, endIdx)
	addGauss(minSigma, minAmplitude, bestX0, dane.lambda[:], minFlux[:], startIdx, endIdx)
	addGauss(maxSigma, maxAmplitude, bestX0, dane.lambda[:], maxFlux[:], startIdx, endIdx)

	saveModelToFile(dane.lambda[:], modelFlux[:], "best", c)
	saveModelToFile(dane.lambda[:], minFlux[:], "min", c)
	saveModelToFile(dane.lambda[:], maxFlux[:], "max", c)

}

addGauss :: proc(sigma, amplitude, x0: f64, lambda, flux: []f64, start, end: int) {

	twoSigSqr := 2 * sigma * sigma
	for i := start; i <= end; i += 1 {
		lambdaDiff := lambda[i] - x0
		exponent := -(lambdaDiff * lambdaDiff) / (twoSigSqr)
		flux[i] += amplitude * math.exp(exponent)
	}
}

findMinMaxParameters :: proc(arr: []result, c: ^config) {

	minAlpha := math.F64_MAX
	maxAlpha := math.F64_MIN

	minAmplitude := math.F64_MAX
	maxAmplitude := math.F64_MIN

	minSigma := math.F64_MAX
	maxSigma := math.F64_MIN

	minX0 := math.F64_MAX
	maxX0 := math.F64_MIN

	for e in arr {
		if e.alpha < minAlpha do minAlpha = e.alpha
		if e.alpha > maxAlpha do maxAlpha = e.alpha

		if e.amplitude < minAmplitude do minAmplitude = e.amplitude
		if e.amplitude > maxAmplitude do maxAmplitude = e.amplitude

		if e.sigma < minSigma do minSigma = e.sigma
		if e.sigma > maxSigma do maxSigma = e.sigma

		if e.x0 < minX0 do minX0 = e.x0
		if e.x0 > maxX0 do maxX0 = e.x0
	}

	c.finalResult.minAlpha = minAlpha
	c.finalResult.maxAlpha = maxAlpha

	c.finalResult.minAmplitude = minAmplitude
	c.finalResult.maxAmplitude = maxAmplitude

	c.finalResult.minSigma = minSigma
	c.finalResult.maxSigma = maxSigma

	c.finalResult.minX0 = minX0
	c.finalResult.maxX0 = maxX0
}

//================== /Stat

//================== File
saveModelToFile :: proc(lambda, flux: []f64, label: string, c: config) {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	fmt.sbprintf(&sb, "# bestAmplitude: %v\n", c.finalResult.amplitude)
	fmt.sbprintf(&sb, "# x0: %v\n", c.finalResult.x0)
	fmt.sbprintf(&sb, "# alpha: %v\n", c.finalResult.alpha)
	fmt.sbprintf(&sb, "# minAmplitude: %v\n", c.finalResult.minAmplitude)
	fmt.sbprintf(&sb, "# maxAmplitude: %v\n", c.finalResult.maxAmplitude)
	fmt.sbprintf(&sb, "# minSigma: %v\n", c.finalResult.minSigma)
	fmt.sbprintf(&sb, "# maxSigma: %v\n", c.finalResult.maxSigma)
	fmt.sbprintf(&sb, "# minAlpha: %v\n", c.finalResult.minAlpha)
	fmt.sbprintf(&sb, "# maxAlpha: %v\n", c.finalResult.maxAlpha)

	fmt.sbprint(&sb, "# Wavelength(A) Flux_density(10^{-16} erg/s^{-1}/cm^{-2}/A^{-1})\n")

	for i in 0 ..< len(lambda) {
		fmt.sbprintf(&sb, "%.7f %e\n", lambda[i], flux[i])
	}

	content := strings.to_string(sb)

	fileName := filepath.base(c.dataPath)
	newName: string = fmt.tprintf("%s_%s", label, fileName)
	path, _ := filepath.join({c.modelsDirPath, newName}, context.temp_allocator)


	ok := os.write_entire_file(path, transmute([]byte)content)

	if ok != nil {
		fmt.eprintfln("Failed to save to file: %s\n", path)
	} else {
		fmt.printf("Saved to file: %s\n", path)
	}
}

saveResultsToFile :: proc(c: config, label: string = "chi2 amplitude sigma x0 alpha") {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	fmt.sbprintf(&sb, "# seed: %v\n", c.seed)
	fmt.sbprintf(&sb, "# N: %f\n", c.numberOfPoints)

	fmt.sbprintf(&sb, "# minAmplitude: %v\n", c.finalResult.minAmplitude)
	fmt.sbprintf(&sb, "# maxAmplitude: %v\n", c.finalResult.maxAmplitude)
	fmt.sbprintf(&sb, "# minSigma: %v\n", c.finalResult.minSigma)
	fmt.sbprintf(&sb, "# maxSigma: %v\n", c.finalResult.maxSigma)
	fmt.sbprintf(&sb, "# minX0: %v\n", c.finalResult.minX0)
	fmt.sbprintf(&sb, "# maxX0: %v\n", c.finalResult.maxX0)
	fmt.sbprintf(&sb, "# minAlpha: %v\n", c.finalResult.minAlpha)
	fmt.sbprintf(&sb, "# maxAlpha: %v\n", c.finalResult.maxAlpha)

	strings.write_string(&sb, "# ")
	strings.write_string(&sb, label)
	strings.write_string(&sb, "\n")

	for i in 0 ..< len(c.results) {
		fmt.sbprintf(
			&sb,
			"%.7f %.7f %.7f %.7f %.7f\n",
			c.results[i].chi2,
			c.results[i].amplitude,
			c.results[i].sigma,
			c.results[i].x0,
			c.results[i].alpha,
		)
	}

	content := strings.to_string(sb)

	fileName := filepath.base(c.dataPath)
	newName: string = fmt.tprintf("wynik_%s", fileName)
	path, _ := filepath.join({c.resultsDirPath, newName}, context.temp_allocator)


	ok := os.write_entire_file(path, transmute([]byte)content)

	if ok != nil {
		fmt.eprintfln("Failed to save to file: %s\n", path)
	} else {
		fmt.printf("Saved to file: %s\n", path)
	}
}

saveResultsToFileJson :: proc(c: config, path: string) {
	saveTemplate :: struct {
		seed:    u64,
		N:       f64,
		results: []result,
	}

	dataToSave: saveTemplate = {
		seed    = c.seed,
		N       = c.numberOfPoints,
		results = c.results[:],
	}

	opt := json.Marshal_Options {
		pretty = true,
	}
	jsonData, jsonErr := json.marshal(dataToSave, opt)

	if jsonErr != nil {
		fmt.eprintfln("JSON Error: %v", jsonErr)
		return
	}

	//defer delete(jsonData)

	ok := os.write_entire_file(path, jsonData)

	if ok != nil {
		fmt.eprintfln("JSON failed to save file: %s", path)
	}
}

loadDataFromFile :: proc(path: string, dane: ^data) {
	file, fErr := os.read_entire_file(path, context.allocator)
	if fErr != nil {
		fmt.println("Failed to read the file")
		return
	}
	defer delete(file)

	fileLines := strings.split_lines(string(file))
	defer delete(fileLines)

	for line in fileLines {
		trimmed := strings.trim_space(line)
		if len(trimmed) == 0 || strings.has_prefix(trimmed, "#") do continue

		columns := strings.fields(trimmed)
		defer delete(columns)

		if len(columns) == 2 {
			lambda, okLambda := strconv.parse_f64(columns[0])
			flux, okFlux := strconv.parse_f64(columns[1])
			if okLambda && okFlux {
				append(&dane.flux, flux)
				append(&dane.lambda, lambda)
			}
		}
	}
}

saveDataToFile :: proc(path: string, dane: ^data, label: string = "X Y") {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	strings.write_string(&sb, "# ")
	strings.write_string(&sb, label)
	strings.write_string(&sb, "\n")

	for i in 0 ..< len(dane.lambda) {
		fmt.sbprintf(&sb, "%f %f\n", dane.lambda[i], dane.flux[i])
	}

	content := strings.to_string(sb)

	ok := os.write_entire_file(path, transmute([]byte)content)

	if ok != nil {
		fmt.eprintfln("Failed to save to file: %s\n", path)
	} else {
		fmt.printf("Saved to file: %s\n", path)
	}
}

//================== /File
