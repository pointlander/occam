// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/pointlander/occam"

	"github.com/pointlander/datum/iris"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func main() {
	// Load the iris data set
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	length := len(fisher)
	for _, value := range fisher {
		sum := 0.0
		for _, measure := range value.Measures {
			sum += measure * measure
		}
		sum = math.Sqrt(sum)
		for i := range value.Measures {
			value.Measures[i] /= sum
		}
	}

	n := occam.NewNetwork(4, length)

	// Set point weights to the iris data
	for i, value := range fisher {
		for j, measure := range value.Measures {
			n.Point.X[4*i+j] = float32(measure)
		}
	}

	entropy := n.GetEntropy(fisher)
	sort.Slice(entropy, func(i, j int) bool {
		return entropy[i].Entropy > entropy[j].Entropy
	})
	for _, e := range entropy {
		fmt.Printf("%.7f %s\n", e.Entropy, e.Label)
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
