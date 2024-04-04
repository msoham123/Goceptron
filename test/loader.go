package main

import (
	"image"
	"image/draw"
	"image/jpeg"
	"io/fs"
	"os"
	"path/filepath"
)

func RegisterJpegFormat() {
	image.RegisterFormat("jpg", "jpg", jpeg.Decode, jpeg.DecodeConfig)
}

func LoadDirImageData(rootDir string, sizeX int, sizeY int,
	imageCap int, data *[][]float64) (count int) {

	filepath.WalkDir(rootDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			panic("File could not be found!")
		}
		if !d.IsDir() {
			file, err := os.Open(path)

			if err != nil {
				panic("File could not be opened!")
			}

			defer file.Close()

			imageData, _, err := image.Decode(file)

			if err != nil {
				panic("File could not be decoded!")
			}

			// rect := imageData.Bounds()
			rect := image.Rect(0, 0, sizeX, sizeY)
			rgba := image.NewRGBA(rect)
			draw.Draw(rgba, rect, imageData, rect.Min, draw.Src)

			var pixels []float64

			for _, pixel := range rgba.Pix {
				floatValue := float64(pixel)
				pixels = append(pixels, floatValue)
			}

			*data = append(*data, pixels)
			count++

			if count == imageCap {
				return filepath.SkipAll
			}
		}

		return nil
	})

	return count
}

func LoadSingleImage(path string, sizeX int, sizeY int) (pixels []float64) {
	file, err := os.Open(path)

	if err != nil {
		panic("File could not be opened!")
	}

	defer file.Close()

	imageData, _, err := image.Decode(file)

	if err != nil {
		panic("File could not be decoded!")
	}

	rect := image.Rect(0, 0, sizeX, sizeY)
	rgba := image.NewRGBA(rect)
	draw.Draw(rgba, rect, imageData, rect.Min, draw.Src)

	for _, pixel := range rgba.Pix {
		floatValue := float64(pixel)
		pixels = append(pixels, floatValue)
	}

	return pixels
}
