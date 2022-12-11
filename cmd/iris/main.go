// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
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

var (
	// FlagNormalize is the flag to normalize the data
	FlagNormalize = flag.Bool("normalize", false, "normalize the data")
)

func main() {
	flag.Parse()

	// Load the iris data set
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	length := len(fisher)
	if *FlagNormalize {
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
	}

	n := occam.NewNetwork(4, length)

	// Set point weights to the iris data
	for i, value := range fisher {
		for j, measure := range value.Measures {
			n.Point.X[4*i+j] = float32(measure)
		}
	}

	sum := float32(0.0)
	entropy := n.GetEntropy(fisher)
	sort.Slice(entropy, func(i, j int) bool {
		return entropy[i].Entropy > entropy[j].Entropy
	})
	for _, e := range entropy {
		sum += e.Entropy
		fmt.Printf("%.7f %s\n", e.Entropy, e.Label)
	}

	for i := 1; i < 150; i++ {
		a := float32(0.0)
		//n.Point.X = n.Point.X[:i*4]
		//n.Point.S[1] = i
		d := make([]iris.Iris, 0, 150)
		for _, e := range entropy[:i] {
			//for j, measure := range e.Measures {
			//	n.Point.X[4*i+j] = float32(measure)
			//}
			d = append(d, iris.Iris{
				Measures: e.Measures,
				Label:    e.Label,
			})
		}
		_ = d
		e := n.GetEntropy(d)
		for _, e := range e {
			a += e.Entropy
		}

		b := float32(0.0)
		//n.Point.X = n.Point.X[:(150-i)*4]
		//n.Point.S[1] = 150 - i
		d = make([]iris.Iris, 0, 150)
		for _, e := range entropy[i:] {
			//for j, measure := range e.Measures {
			//	n.Point.X[4*i+j] = float32(measure)
			//}
			d = append(d, iris.Iris{
				Measures: e.Measures,
				Label:    e.Label,
			})
		}
		_ = d
		e = n.GetEntropy(d)
		for _, e := range e {
			b += e.Entropy
		}
		fmt.Println(i, sum, a, b, a+b, sum-(a+b))
	}

	n = occam.NewNetwork(4, length)

	// Set point weights to the iris data
	for i, value := range fisher {
		for j, measure := range value.Measures {
			n.Point.X[4*i+j] = float32(measure)
		}
	}

	// The stochastic gradient descent loop
	epochs := 8 * 1024
	if *FlagNormalize {
		epochs = 1024
	}
	for n.I < epochs {
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
