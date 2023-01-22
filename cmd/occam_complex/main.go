// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"time"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tc128"

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

// SphericalSoftmax is the spherical softmax function
// https://arxiv.org/abs/1511.05042
func SphericalSoftmax(k tc128.Continuation, node int, a *tc128.V, options ...map[string]interface{}) bool {
	const E = complex128(0)
	c, size, width := tc128.NewV(a.S...), len(a.X), a.S[0]
	values, sums, row := make([]complex128, width), make([]complex128, a.S[1]), 0
	for i := 0; i < size; i += width {
		sum := complex128(0.0)
		for j, ax := range a.X[i : i+width] {
			values[j] = ax*ax + E
			sum += values[j]
		}
		for _, cx := range values {
			c.X = append(c.X, (cx+E)/sum)
		}
		sums[row] = sum
		row++
	}
	if k(&c) {
		return true
	}
	// (2 a (b^2 + c^2 + d^2 + 0.003))/(a^2 + b^2 + c^2 + d^2 + 0.004)^2
	for i, d := range c.D {
		ax, sum := a.X[i], sums[i/width]
		//a.D[i] += d*(2*ax*(sum-(ax*ax+E)))/(sum*sum) - d*cx*2*ax/sum
		a.D[i] += d * (2 * ax * (sum - (ax*ax + E))) / (sum * sum)
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

	// Load the iris data set
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	width, length := 4, len(fisher)

	set := tc128.NewSet()
	set.Add("points", width, length)
	points := set.ByName["points"]
	for _, value := range fisher {
		measures := value.Measures
		points.X = append(points.X,
			cmplx.Rect(measures[0], rnd.Float64()*math.Pi),
			cmplx.Rect(measures[1], rnd.Float64()*math.Pi),
			cmplx.Rect(measures[2], rnd.Float64()*math.Pi),
			cmplx.Rect(measures[3], rnd.Float64()*math.Pi),
		)
	}
	points.States = make([][]complex128, StateTotal)
	for i := range points.States {
		points.States[i] = make([]complex128, len(points.X))
	}

	softmax := tc128.U(SphericalSoftmax)
	l1 := softmax(tc128.Mul(set.Get("points"), set.Get("points")))
	l2 := softmax(tc128.T(tc128.Mul(l1, tc128.T(set.Get("points")))))
	cost := tc128.Entropy(l2)

	loss := make(plotter.XYs, 0, 8)

	i := 1
	pow := func(x complex128) complex128 {
		return cmplx.Pow(x, complex(float64(i), 0))
	}

	// The stochastic gradient descent loop
	for i < 1024 {
		start := time.Now()
		// Calculate the gradients
		total := tc128.Gradient(cost).X[0]

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
				set.Weights[j].X[k] -= Eta * mhat / (cmplx.Sqrt(vhat) + 1e-8)
			}
		}

		// Housekeeping
		end := time.Since(start)
		fmt.Println(i, cmplx.Abs(total), end)
		set.Zero()

		if cmplx.IsNaN(total) {
			fmt.Println(total)
			break
		}

		loss = append(loss, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		i++
	}

	// Plot the cost
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(loss)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "occam_complex_cost.png")
	if err != nil {
		panic(err)
	}

	set.Save("occam_complex_set.w", 0, 0)
}
