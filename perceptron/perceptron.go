package perceptron

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Perceptron struct {
	// Learning rate of Perceptron
	alpha float64

	// Perceptron weights
	weights []float64
}

func CreatePerceptron(featureNum int, alpha float64) (p *Perceptron) {

	// Create new Perceptron and initialize fields
	p = new(Perceptron)
	p.alpha = alpha

	// size of weights accounts for each feature plus bias
	p.weights = make([]float64, featureNum+1)

	// Generate rand with time-sensitive seed for best randomness
	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	/* Using norm of weight size, generate normalized weights
	for faster convergence later on */
	norm := math.Sqrt(float64(len(p.weights)))
	for i := 0; i < featureNum+1; i++ {
		p.weights[i] = r.NormFloat64() / norm
	}

	return p
}

func activationFunction(dot float64) int {
	/* If the net input is greater than 0, then return 1
	Serves as a step function */
	if dot > 0 {
		return 1
	}

	return 0
}

func dotProduct(a []float64, b []float64) (result float64) {
	// Length checking, else throw error
	if len(a) != len(b) {
		fmt.Printf("a is %d and b is %d\n", len(a), len(b))
		panic("dotProduct: Input vectors a and b do not have the same length!")
	}

	// Calculate sum of element-wise product
	for i := range a {
		result += a[i] * b[i]
	}

	return result
}

func Train(p *Perceptron, trainingSet [][]float64, targets []int, epochs int) {

	/* Augment training data with extra columns to
	represent bias as trainable parameter*/

	trainingCopy := make([][]float64, len(trainingSet))
	copy(trainingCopy, trainingSet)

	for i := range trainingCopy {
		trainingCopy[i] = append(trainingCopy[i], 1)
	}

	fmt.Print("[GOCEPTRON]: Beginning training of perceptron...\n\n")

	failures := make([]int, epochs)

	// Loop over each epoch e for training
	for e := 0; e < epochs; e++ {

		// Loop over each data row
		for i := range trainingCopy {
			pred := activationFunction(dotProduct(trainingCopy[i], p.weights))

			/*If prediction does not match target,
			then perform weight update*/
			if pred != targets[i] {

				failures[e]++

				// Calculate the error
				err := pred - targets[i]

				// Update the weight matrix using gradient descent
				for w := range p.weights {
					p.weights[w] -= float64(err) * p.alpha * trainingCopy[i][w]
				}
			}

		}

		failureRate := (float64(failures[e]) / float64(len(trainingCopy))) * 100

		fmt.Printf("\t> At epoch %d, failure rate was %f percent...\n", e, failureRate)

		if failureRate == 0 {
			epochs = e
			break
		}
	}

	fmt.Printf("\n[GOCEPTRON]: Finished training of perceptron in %d epochs...\n\n", epochs)
}

func Predict(p *Perceptron, data []float64) (pred int) {
	/* Augment data point with extra column to
	represent bias as trainable parameter*/
	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)

	dataCopy = append(dataCopy, 1)

	// Length checking, else throw error
	if len(dataCopy) != len(p.weights) {
		panic("Predict: data and weights must have same length!")
	}

	// Take dot of input data and weights and apply activation function
	pred = activationFunction(dotProduct(dataCopy, p.weights))

	return pred
}

func PredictOnSet(p *Perceptron, testSet [][]float64, targets []int) (pred []int) {
	// Length checking, else throw error
	if len(testSet) != len(targets) {
		panic("PredictOnSet: testSet and targets must have same length!")
	}

	fmt.Print("[GOCEPTRON]: Beginning prediction using perceptron...\n")

	// Take dot of input data and weights and apply activation function
	pred = make([]int, len(testSet))

	count := 0
	for i := range testSet {
		pred[i] = Predict(p, testSet[i])

		// fmt.Printf("\t> data = {%v}, target = %d, pred = %d\n", testSet[i], targets[i], pred[i])

		if pred[i] == targets[i] {
			count++
		}
	}

	// Print success stats
	fmt.Printf("\n[GOCEPTRON] Accuracy of run was %f percent\n", (float64(count) / float64(len(testSet)) * 100))

	return pred
}
