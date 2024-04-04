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
		case "-dogs":
			dogTest()
		case "-allBinary":
			orTest()
			andTest()
			xorTest()
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

	/* NOTE: XOR is non linear so the perceptron can't actually learn this
	without more layers and nonlinear activation functions (deep learning)
	*/
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

/*
This dataset is from:
https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
Make sure you download it to data/ as cats_dogs (data/cats_dogs/) in order for this test to run!
1 represents dogs
*/
func dogTest() {

	// --CONFIG--

	// Epochs for training
	epochs := 500

	/* Alpha value in gradient descent 
	controls how large (or small) we move
	so is learning rate*/
	alpha := 0.001

	/* Values to resize the image by
	Making this smaller will allow 
	for faster training runs at cost of accuracy
	*/
	sizeX, sizeY := 500, 500

	// How many images to use for training
	imageCap := 200
	// -----------

	fmt.Println("Running Dog Perceptron test...")

	RegisterJpegFormat()

	rootDir := "./data/cats_dogs/train/dogs/"

	var trainingData [][]float64
	var targets []int

	fmt.Println("Loading Dog Images")

	count := LoadDirImageData(rootDir, sizeX, sizeY, imageCap, &trainingData)

	for i := 0; i < count; i++ {
		targets = append(targets, 1)
	}

	fmt.Printf("Loaded %d training dog images...\n", count)

	fmt.Println("Loading Cat Images")

	rootDir = "./data/cats_dogs/train/cats/"

	count = LoadDirImageData(rootDir, sizeX, sizeY, imageCap, &trainingData)

	for i := 0; i < count; i++ {
		targets = append(targets, 0)
	}

	fmt.Printf("Loaded %d training cat images...\n", count)

	p := perceptron.CreatePerceptron(len(trainingData[0]), alpha)

	perceptron.Train(p, trainingData, targets, epochs)

	var testingData [][]float64
	var truths []int

	rootDir = "./data/cats_dogs/test/dogs/"

	count = LoadDirImageData(rootDir, sizeX, sizeY, 500, &testingData)

	for i := 0; i < count; i++ {
		truths = append(truths, 1)
	}

	fmt.Printf("Loaded %d testing dog images...\n", count)

	fmt.Println("Loading Cat Images")

	rootDir = "./data/cats_dogs/test/cats/"

	count = LoadDirImageData(rootDir, sizeX, sizeY, 500, &testingData)

	for i := 0; i < count; i++ {
		truths = append(truths, 0)
	}

	fmt.Printf("Loaded %d testing cat images...\n", count)

	perceptron.PredictOnSet(p, testingData, truths)
}

