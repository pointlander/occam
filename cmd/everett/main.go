// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/pointlander/gradient/tf32"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
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
	// Eta is the learning rate
	Eta = .001
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
func Softmax(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
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

var (
	//FlagInfer inference mode
	FlagInfer = flag.String("infer", "", "inference mode")
	//FlagTrain train mode
	FlagTrain = flag.String("train", "en", "train mode")
)

func main() {
	flag.Parse()
	rnd := rand.New(rand.NewSource(1))

	width := 8
	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}

	// Create the weight data matrix
	set := tf32.NewSet()
	set.Add("points", width, 1024)
	for _, w := range set.Weights {
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32((2*rnd.Float64() - 1)))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	// The neural network is the attention model from attention is all you need
	softmax := tf32.U(Softmax)
	l1 := softmax(tf32.Mul(set.Get("points"), set.Get("points")))
	l2 := softmax(tf32.Mul(tf32.T(set.Get("points")), l1))
	cost := tf32.Entropy(l2)

	points := make(plotter.XYs, 0, 8)
	// The stochastic gradient descent loop
	for i < 4*1024 {
		start := time.Now()
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
				set.Weights[j].X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}

		// Housekeeping
		end := time.Since(start)
		fmt.Println(i, total, end)
		set.Zero()

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

	particles := set.ByName["points"].X
	for i := 0; i < 1024; i++ {
		for j := 0; j < width; j++ {
			fmt.Printf("%f ", particles[i*width+j])
		}
		fmt.Println()
	}

	particles64 := make([]float64, 0, len(particles))
	for _, v := range particles {
		particles64 = append(particles64, float64(v))
	}
	ranks := mat.NewDense(1024, width, particles64)
	var pc stat.PC
	ok := pc.PrincipalComponents(ranks, nil)
	if !ok {
		panic("PrincipalComponents failed")
	}
	k := 2
	var proj mat.Dense
	var vec mat.Dense
	pc.VectorsTo(&vec)
	proj.Mul(ranks, vec.Slice(0, width, 0, k))

	fmt.Printf("\n")
	points = make(plotter.XYs, 0, 8)
	for i := 0; i < 1024; i++ {
		fmt.Println(proj.At(i, 0), proj.At(i, 1))
		points = append(points, plotter.XY{X: proj.At(i, 0), Y: proj.At(i, 1)})
	}

	p = plot.New()

	p.Title.Text = "x vs y"
	p.X.Label.Text = "x"
	p.Y.Label.Text = "y"

	scatter, err = plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(3)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "projection.png")
	if err != nil {
		panic(err)
	}
}
