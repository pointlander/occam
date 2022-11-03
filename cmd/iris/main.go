// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/occam"

	"github.com/pointlander/datum/iris"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var (
	FlagRNN = flag.Bool("rnn", false, "rnn mode")
)

func main() {
	flag.Parse()

	if *FlagRNN {
		n := occam.NewNetwork(8, 8)
		state := make([]float64, 8)
		for i := 0; i < 8; i++ {
			for j := 0; j < 8; j++ {
				n.Point.X[8*i+j] = n.Rnd.Float32()
			}
		}

		state[0] = 1
		for i := 0; i < 8*1024; i++ {
			total := n.Iterate(state)

			if math.IsNaN(float64(total)) {
				fmt.Println(total)
				break
			}

			n.L2(func(a *tf32.V) bool {
				for i, value := range a.X {
					state[i] = float64(value)
				}
				return true
			})
			fmt.Println(state)
			state[0] = n.Rnd.Float64()
			state[2] = n.Rnd.Float64()
		}
		for i := 0; i < 8; i++ {
			for j := 0; j < 8; j++ {
				fmt.Printf("%f ", n.Point.X[i*8+j])
			}
			fmt.Printf("\n")
		}

		// Plot the cost
		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(n.Points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
		if err != nil {
			panic(err)
		}
		return
	}

	// Load the iris data set
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	length := len(fisher)

	n := occam.NewNetwork(4, length)

	// Set point weights to the iris data
	for i, value := range fisher {
		for j, measure := range value.Measures {
			n.Point.X[4*i+j] = float32(measure)
		}
	}

	// The stochastic gradient descent loop
	for n.I < 8*1024 {
		// Randomly select a load the input
		index := n.Rnd.Intn(length)
		sample := fisher[index]
		total := n.Iterate(sample.Measures)

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}
	}

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(n.Points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	n.Set.Save("set.w", 0, 0)

	n.Analyzer(fisher)
}
