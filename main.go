// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
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

	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	max := 0.0
	for _, value := range fisher {
		for _, measure := range value.Measures {
			if measure < 0 {
				measure = -measure
			}
			if measure > max {
				max = measure
			}
		}
	}

	others := tf32.NewSet()
	others.Add("input", 4, 1)
	input := others.ByName["input"]
	input.X = input.X[:cap(input.X)]

	set := tf32.NewSet()
	set.Add("points", 4, 150)

	for _, w := range set.Weights {
		if strings.HasPrefix(w.N, "b") {
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

	softmax := tf32.U(Softmax)
	l1 := tf32.Mul(set.Get("points"), others.Get("input"))
	out := softmax(l1)
	cost := tf32.Sum(tf32.Entropy(out))

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
	for i < 1024 {
		index := rnd.Intn(len(fisher))
		sample := fisher[index]
		for i, measure := range sample.Measures {
			input.X[i] = float32(measure / max)
		}
		total := tf32.Gradient(cost).X[0]

		b1, b2 := pow(B1), pow(B2)
		for j, w := range set.Weights {
			for k, d := range w.D {
				g := d / float32(len(fisher))
				m := B1*w.States[StateM][k] + (1-B1)*g
				v := B2*w.States[StateV][k] + (1-B2)*g*g
				w.States[StateM][k] = m
				w.States[StateV][k] = v
				mhat := m / (1 - b1)
				vhat := v / (1 - b2)
				set.Weights[j].X[k] -= eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
			}
		}

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

	for i := 0; i < len(fisher); i++ {
		sample := fisher[i]
		for i, measure := range sample.Measures {
			input.X[i] = float32(measure / max)
		}
		out(func(a *tf32.V) bool {
			max, index := float32(0.0), 0
			for j, value := range a.X {
				if value > max {
					max, index = value, j
				}
			}
			fmt.Println(index, fisher[i].Label)
			return true
		})
	}

}
