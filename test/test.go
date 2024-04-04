package main

import (
	"fmt"
	"goceptron/perceptron"
	"os"
)

func main() {

	argsWithoutProg := os.Args[1:]

	for i := range argsWithoutProg {
		arg := argsWithoutProg[i]

		switch arg {
		case "-or":
			orTest()
		case "-and":
			andTest()
		case "-xor":
			xorTest()
		case "-not":
			notTest()
		default:
			fmt.Printf("[GOCEPTRON]: Invalid argument \"%s\"\n", arg)
		}

	}
}

func orTest() {

	fmt.Println("Running OR Perceptron test...")

	trainingData := [][]float64{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
	}
	targets := []int{0, 1, 1, 1}
	epochs := 30
	alpha := 0.1

	p := perceptron.CreatePerceptron(len(trainingData[0]), alpha)

	perceptron.Train(p, trainingData, targets, epochs)

	perceptron.PredictOnSet(p, trainingData, targets)
}

func andTest() {

	fmt.Println("Running AND Perceptron test...")

	trainingData := [][]float64{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
	}
	targets := []int{0, 0, 0, 1}
	epochs := 30
	alpha := 0.1

	p := perceptron.CreatePerceptron(len(trainingData[0]), alpha)

	perceptron.Train(p, trainingData, targets, epochs)

	perceptron.PredictOnSet(p, trainingData, targets)
}

func xorTest() {

	// NOTE: XOR is non linear so the perceptron can't actually learn this
	fmt.Println("Running XOR Perceptron test...")

	trainingData := [][]float64{
		{0, 0},
		{1, 0},
		{0, 1},
		{1, 1},
	}
	targets := []int{0, 1, 1, 0}
	epochs := 30
	alpha := 0.1

	p := perceptron.CreatePerceptron(len(trainingData[0]), alpha)

	perceptron.Train(p, trainingData, targets, epochs)

	perceptron.PredictOnSet(p, trainingData, targets)
}

func notTest() {

	fmt.Println("Running AND Perceptron test...")

	trainingData := [][]float64{
		{0},
		{1},
	}
	targets := []int{1, 0}
	epochs := 30
	alpha := 0.1

	p := perceptron.CreatePerceptron(len(trainingData[0]), alpha)

	perceptron.Train(p, trainingData, targets, epochs)

	perceptron.PredictOnSet(p, trainingData, targets)
}