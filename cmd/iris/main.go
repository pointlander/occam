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

	entropy := n.GetEntropy(fisher)
	sort.Slice(entropy, func(i, j int) bool {
		return entropy[i].Entropy > entropy[j].Entropy
	})
	last := float32(0.0)
	for i, e := range entropy {
		entropy[i].Order = i
		fmt.Printf("%3d %.7f %.7f %s\n", i, e.Entropy, last-e.Entropy, e.Label)
		last = e.Entropy
	}
	gradients := n.GetGradients(fisher)
	vectors := n.GetVectors2(fisher)

	for i, grad := range gradients {
		xy, x, y, x2, y2 := float32(0.0), float32(0.0), float32(0.0), float32(0.0), float32(0.0)
		for i, grad := range grad {
			grad = -grad
			measure := float32(vectors[i].Measures[i])
			xy += grad * measure
			x += grad
			y += measure
			x2 += grad * grad
			y2 += measure * measure
		}
		xy /= float32(len(gradients))
		x /= float32(len(gradients))
		y /= float32(len(gradients))
		x2 /= float32(len(gradients))
		y2 /= float32(len(gradients))
		fmt.Println(i, (xy-x*y)/(float32(math.Sqrt(float64(x2-x*x)))*float32(math.Sqrt(float64(y2-y*y)))), grad, vectors[i].Measures)
	}

	var split func(n *occam.Network, depth int, entropy []occam.Entropy, splits []int) []int
	split = func(n *occam.Network, depth int, entropy []occam.Entropy, splits []int) []int {
		if depth == 0 {
			return splits
		}

		sum := float32(0.0)
		for _, e := range entropy {
			sum += e.Entropy
		}
		avg, vari := sum/float32(len(entropy)), float32(0.0)
		for _, e := range entropy {
			difference := e.Entropy - avg
			vari += difference * difference
		}
		vari /= float32(len(entropy))

		index, max := 0, float32(0.0)
		for i := 1; i < len(entropy); i++ {
			suma, counta := float32(0.0), float32(0.0)
			for _, e := range entropy[:i] {
				suma += e.Entropy
				counta++
			}
			avga, varia := suma/counta, float32(0.0)
			for _, e := range entropy[:i] {
				difference := e.Entropy - avga
				varia += difference * difference
			}
			varia /= counta

			sumb, countb := float32(0.0), float32(0.0)
			for _, e := range entropy[i:] {
				sumb += e.Entropy
				countb++
			}
			avgb, varib := sumb/countb, float32(0.0)
			for _, e := range entropy[i:] {
				difference := e.Entropy - avgb
				varib += difference * difference
			}
			varib /= countb

			gain := vari - (varia + varib)
			if gain > max {
				index, max = i, gain
			}
		}
		splits = append(splits, entropy[index].Order)

		/*dat := make([]iris.Iris, 0, 8)
		for _, e := range entropy[index:] {
			dat = append(dat, iris.Iris{
				Measures: e.Measures,
				Label:    e.Label,
			})
		}*/
		splits = split(n, depth-1, entropy[index:], splits)

		/*dat = make([]iris.Iris, 0, 8)
		for _, e := range entropy[:index] {
			dat = append(dat, iris.Iris{
				Measures: e.Measures,
				Label:    e.Label,
			})
		}*/
		return split(n, depth-1, entropy[:index], splits)
	}
	splits := split(n, 2, entropy, []int{})
	fmt.Println(splits)

	nonlinear := make([]occam.Entropy, splits[0])
	copy(nonlinear, entropy[:splits[0]])
	for i := range nonlinear {
		nonlinear[i].Index = i
	}
	type AB struct {
		A, B uint
	}
	ab := make([]AB, len(nonlinear))
	for i := 0; i < 1024; i++ {
		data := make([]iris.Iris, 0, 8)
		for _, e := range nonlinear {
			measures := make([]float64, len(e.Measures))
			for j, m := range e.Measures {
				measures[j] = m + n.Rnd.NormFloat64()*0.1
			}
			data = append(data, iris.Iris{
				Measures: measures,
				Label:    e.Label,
			})
		}

		n := occam.NewNetwork(4, len(data))
		// Set point weights to the iris data
		for i, value := range data {
			for j, measure := range value.Measures {
				n.Point.X[4*i+j] = float32(measure)
			}
		}

		entropy := n.GetEntropy(data)
		sort.Slice(entropy, func(i, j int) bool {
			return entropy[i].Entropy > entropy[j].Entropy
		})
		for j := range entropy {
			entropy[j].Order = j
		}
		splits := split(n, 1, entropy, []int{})
		for j, e := range entropy {
			if j < splits[0] {
				ab[e.Index].A++
			} else {
				ab[e.Index].B++
			}
		}
	}
	for i, e := range nonlinear {
		fmt.Println(i, e.Label, ab[e.Index].A, ab[e.Index].B)
	}

	data := make([]iris.Iris, 0, 8)
	for _, e := range nonlinear {
		measures := make([]float64, len(e.Measures)+2)
		copy(measures, e.Measures)
		measures[len(e.Measures)] = float64(ab[e.Index].A) / 1024
		measures[len(e.Measures)+1] = float64(ab[e.Index].B) / 1024
		data = append(data, iris.Iris{
			Measures: measures,
			Label:    e.Label,
		})
	}
	{
		n := occam.NewNetwork(6, len(data))
		// Set point weights to the iris data
		for i, value := range data {
			for j, measure := range value.Measures {
				n.Point.X[6*i+j] = float32(measure)
			}
		}

		entropy := n.GetEntropy(data)
		sort.Slice(entropy, func(i, j int) bool {
			return entropy[i].Entropy > entropy[j].Entropy
		})
		for j := range entropy {
			entropy[j].Order = j
		}
		splits := split(n, 1, entropy, []int{})
		for i, e := range entropy {
			fmt.Printf("%3d %.7f %s\n", i, e.Entropy, e.Label)
		}
		fmt.Println(splits)
	}

	// The stochastic gradient descent loop
	epochs := 8 * 1024
	if *FlagNormalize {
		epochs = 1024
	}
	indexes := make([]int, length)
	for i := range indexes {
		indexes[i] = i
	}
	for n.I < epochs {
		n.Rnd.Shuffle(length, func(i, j int) {
			indexes[i], indexes[j] = indexes[j], indexes[i]
		})
		for _, index := range indexes {
			// Randomly select a load the input
			sample := fisher[index]
			total := n.Iterate(sample.Measures)

			if math.IsNaN(float64(total)) {
				fmt.Println(total)
				break
			}
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

	entropy2 := n.GetEntropy(fisher)
	for i, e := range entropy {
		entropy[i].Optimized = entropy2[e.Index].Entropy
	}
	sort.Slice(entropy, func(i, j int) bool {
		return (entropy[i].Entropy - entropy[i].Optimized) > (entropy[j].Entropy - entropy[j].Optimized)
	})
	for i, e := range entropy {
		fmt.Printf("%3d %.7f %.7f %.7f %s\n", i, e.Entropy, e.Optimized, e.Entropy-e.Optimized, e.Label)
		entropy[i].Entropy = e.Entropy - e.Optimized
	}
	splits2 := split(n, 2, entropy, []int{})
	fmt.Println(splits2)
}
