package main

import "core:encoding/json"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

data :: struct {
	lambda: [dynamic]f64,
	flux:   [dynamic]f64,
}
config :: struct {
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
	finalResult:       result,
	results:           [dynamic]result,
}

result :: struct {
	chi2:      f64,
	amplitude: f64,
	sigma:     f64,
	x0:        f64,
	alpha:     f64,
}

main :: proc() {

	konfiguracja: config

	konfiguracja.windowX = 2801
	konfiguracja.windowWidth = 50
	konfiguracja.chiThreshold = 2
	konfiguracja.parametrCount = 4

	konfiguracja.contLambda0 = 3000
	konfiguracja.contB = 2.128
	konfiguracja.contAlphaGuess = -1.556
	konfiguracja.contAlphaWidth = 0.01

	konfiguracja.amplitudeGuess = 1
	konfiguracja.amplitudeWidth = 3
	konfiguracja.x0Guess = 2800 // NOTE: czy trzeba uzglednic jakos zakres okna z zakresem srodka krzywej?
	konfiguracja.x0Width = 50
	konfiguracja.sigmaGuess = 16
	konfiguracja.sigmaWidth = 2

	konfiguracja.iterationCount = 1_000_000
	konfiguracja.percentOfBestChi2 = 0.01
	length := f64(konfiguracja.iterationCount) * konfiguracja.percentOfBestChi2

	konfiguracja.results = make([dynamic]result, i32(length), i32(length))

	for &w in konfiguracja.results {
		// NOTE: musimy dodac tu max f64 bo inaczej nie zapiszemy wynikow do tej tablicy,
		//powiewaz domyslnie pamiec w Odnie jest zerowana a chi^2 nie bedzie mniejsze od zera
		w.chi2 = math.F64_MAX
	}

	dane: data
	dane.flux = make([dynamic]f64)
	dane.lambda = make([dynamic]f64)

	pathTest := "testModel.txt"

	setSeedOfThePRNG(&konfiguracja)

	loadDataFromFile(pathTest, &dane)
	monteCarlo(&konfiguracja, dane)

	fmt.printf("Results: %#v\n", konfiguracja.finalResult)

	saveResultsToFileJson(konfiguracja, "wynik.json")
	saveResultsToFile(konfiguracja, "wynikCSV.txt")


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


}

monteCarlo :: proc(c: ^config, dane: data) {
	best: result
	best.chi2 = math.F64_MAX
	oneOverSqrt2Pi: f64 = 1 / math.sqrt_f64(2 * math.PI)

	for j in 0 ..< len(dane.lambda) {
		lambda := dane.lambda[j]
		if lambda >= c.windowX - c.windowWidth && lambda <= c.windowX + c.windowWidth {
			c.numberOfPoints += 1
		}
	}

	for i in 0 ..< c.iterationCount {

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

		for j in 0 ..< len(dane.lambda) {
			lambda := dane.lambda[j]

			if lambda < c.windowX - c.windowWidth || lambda > c.windowX + c.windowWidth do continue // NOTE: Tu sprawdamy czy znajdujemy się na odpowiedniej dlugosci fali do zaobserwowania danej linii emisyjnej


			// NOTE: B * (lambda/lambda0)^-alpha
			cont := c.contB * math.pow(lambda / c.contLambda0, temp.alpha) // FIX: czy przy temp.alpha musi byc -, z minusem wychodzi źle
			tempFlux := dane.flux[j] - cont


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
	}
	c.finalResult = best
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

//================== /Stat

//================== File
saveResultsToFile :: proc(
	c: config,
	path: string,
	label: string = "chi2 amplitude sigma x0 alpha",
) {
	sb := strings.builder_make()
	defer strings.builder_destroy(&sb)

	fmt.sbprintf(&sb, "# seed: %v\n", c.seed)
	fmt.sbprintf(&sb, "# N: %f\n", c.numberOfPoints)

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
