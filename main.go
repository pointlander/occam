// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.9
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.999
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Softmax is the softmax function for big numbers
func Softmax(k tf32.Continuation, node int, a *tf32.V) bool {
	c, size, width := tf32.NewV(a.S...), len(a.X), a.S[0]
	max := float32(0)
	for _, v := range a.X {
		if v > max {
			max = v
		}
	}
	values := make([]float64, width)
	for i := 0; i < size; i += width {
		s := float64(max) * S
		sum := 0.0
		for j, ax := range a.X[i : i+width] {
			values[j] = math.Exp(float64(ax) - s)
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, float32(cx/sum))
		}
	}
	if k(&c) {
		return true
	}
	for i, d := range c.D {
		cx := c.X[i]
		a.D[i] += d * (cx - cx*cx)
	}
	return false
}

func main() {
	rnd := rand.New(rand.NewSource(1))

	// Load the iris data set
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	length := len(fisher)

	// Create the input data matrix
	others := tf32.NewSet()
	others.Add("input", 4, 1)
	input := others.ByName["input"]
	input.X = input.X[:cap(input.X)]

	// Create the weight data matrix
	set := tf32.NewSet()
	set.Add("points", 4, 150)
	point := set.ByName["points"]

	// Initialize the weights
	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") || strings.HasPrefix(w.N, "points") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rnd.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	// Set point weights to the iris data
	for i, value := range fisher {
		for j, measure := range value.Measures {
			point.X[4*i+j] = float32(measure)
		}
	}

	// The neural network is the attention model from attention is all you need
	softmax := tf32.U(Softmax)
	l1 := softmax(tf32.Mul(set.Get("points"), others.Get("input")))
	l2 := softmax(tf32.Mul(tf32.T(set.Get("points")), l1))
	cost := tf32.Entropy(l2)

	// Initialize the stochastic gradient descent loop
	i, start := 1, time.Now()
	eta := float32(.001)
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}
	points := make(plotter.XYs, 0, 8)
	// The stochastic gradient descent loop
	for i < 8*1024 {
		// Randomly select a load the input
		index := rnd.Intn(length)
		sample := fisher[index]
		for i, measure := range sample.Measures {
			input.X[i] = float32(measure)
		}
		// Calculate the gradients
		total := tf32.Gradient(cost).X[0]

		// Update the point weights with the partial derivatives using adam
		b1, b2 := pow(B1), pow(B2)
		for j, w := range set.Weights {
			for k, d := range w.D {
				g := d
				m := B1*w.States[StateM][k] + (1-B1)*g
				v := B2*w.States[StateV][k] + (1-B2)*g*g
				w.States[StateM][k] = m
				w.States[StateV][k] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				set.Weights[j].X[k] -= eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}

		// Housekeeping
		end := time.Since(start)
		fmt.Println(i, total, end)
		set.Zero()
		others.Zero()
		start = time.Now()

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}
		points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
		i++
	}

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
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

	set.Save("set.w", 0, 0)

	// For each input, label and sort the points in terms of distance to the input
	type Point struct {
		Index int
		Rank  float32
	}
	type Input struct {
		Points []Point
		Label  string
	}
	inputs := make([]Input, 0, length)
	for i := 0; i < length; i++ {
		// Load the input
		sample := fisher[i]
		for i, measure := range sample.Measures {
			input.X[i] = float32(measure)
		}
		// Calculate the l1 output of the neural network
		l1(func(a *tf32.V) bool {
			points := make([]Point, 0, length)
			for j, value := range a.X {
				points = append(points, Point{
					Index: j,
					Rank:  value,
				})
			}
			sort.Slice(points, func(i, j int) bool {
				return points[i].Rank > points[j].Rank
			})
			inputs = append(inputs, Input{
				Points: points,
				Label:  fisher[i].Label,
			})
			return true
		})
	}
	// Sort the inputs by the point indexes
	sort.Slice(inputs, func(i, j int) bool {
		index := 0
		for inputs[i].Points[index].Index == inputs[j].Points[index].Index {
			index++
			if index == length {
				return false
			}
		}
		return inputs[i].Points[index].Index < inputs[j].Points[index].Index
	})
	// Count how many inputs have the same label as their nearest neighbor
	same := 0
	for i, label := range inputs {
		max, index := 0, 0
		for j, l := range inputs {
			if i == j {
				continue
			}
			total := 0
			for k, value := range label.Points {
				a, b := value.Index, l.Points[k].Index
				if a == b {
					total++
				}
			}
			if total > max {
				max, index = total, j
			}
		}
		for _, rank := range label.Points[:18] {
			fmt.Printf("%03d ", rank.Index)
		}
		fmt.Println(label.Label, inputs[index].Label)
		if label.Label == inputs[index].Label {
			same++
		}
	}
	fmt.Println(same, length, float64(same)/float64(length))
}
