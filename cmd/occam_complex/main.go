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
	"sort"
	"time"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/occam"

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
	Eta = .1
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
	_ = rnd

	// Load the iris data set
	datum, err := iris.Load()
	if err != nil {
		panic(err)
	}
	fisher := datum.Fisher
	width, length := 4, len(fisher)

	averages := make([]float64, width)
	for _, value := range fisher {
		measures := value.Measures
		for i, value := range measures {
			averages[i] += value
		}
	}
	for i, value := range averages {
		averages[i] = value / float64(length)
	}

	others := tc128.NewSet()
	others.Add("inputs", width, length)
	inputs := others.ByName["inputs"]
	for _, value := range fisher {
		measures := value.Measures
		inputs.X = append(inputs.X, /*cmplx.Rect(measures[0], rnd.Float64()*2*math.Pi),
			cmplx.Rect(measures[1], rnd.Float64()*2*math.Pi),
			cmplx.Rect(measures[2], rnd.Float64()*2*math.Pi),
			cmplx.Rect(measures[3], rnd.Float64()*2*math.Pi),*/
			complex(measures[0], 0),
			complex(measures[1], 0),
			complex(measures[2], 0),
			complex(measures[3], 0),
			/*complex(measures[0], measures[0]),
			complex(measures[1], measures[1]),
			complex(measures[2], measures[2]),
			complex(measures[3], measures[3]),*/
			/*complex(measures[0], measures[0]-averages[0]),
			complex(measures[1], measures[1]-averages[1]),
			complex(measures[2], measures[2]-averages[2]),
			complex(measures[3], measures[3]-averages[3]),*/
		)
	}

	set := tc128.NewSet()
	set.Add("weights", width, length)
	weights := set.ByName["weights"]
	for _, value := range fisher {
		measures := value.Measures
		weights.X = append(weights.X,
			/*cmplx.Rect(measures[0], rnd.Float64()*2*math.Pi),
			cmplx.Rect(measures[1], rnd.Float64()*2*math.Pi),
			cmplx.Rect(measures[2], rnd.Float64()*2*math.Pi),
			cmplx.Rect(measures[3], rnd.Float64()*2*math.Pi),*/
			complex(measures[0], 0),
			complex(measures[1], 0),
			complex(measures[2], 0),
			complex(measures[3], 0),
			/*complex(measures[0], measures[0]),
			complex(measures[1], measures[1]),
			complex(measures[2], measures[2]),
			complex(measures[3], measures[3]),*/
			/*complex(measures[0], measures[0]-averages[0]),
			complex(measures[1], measures[1]-averages[1]),
			complex(measures[2], measures[2]-averages[2]),
			complex(measures[3], measures[3]-averages[3]),*/
		)
	}
	weights.States = make([][]complex128, StateTotal)
	for i := range weights.States {
		weights.States[i] = make([]complex128, len(weights.X))
	}

	softmax := tc128.U(SphericalSoftmax)
	l1 := softmax(tc128.Mul(set.Get("weights"), others.Get("inputs")))
	l2 := softmax(tc128.T(tc128.Mul(l1, tc128.T(set.Get("weights")))))
	entropy := tc128.Entropy(l2)
	cost := tc128.Avg(entropy)

	type Item struct {
		Label string
		Rank  complex128
	}
	rank := func() {
		items := make([]Item, 0, 8)
		entropy(func(a *tc128.V) bool {
			for i := 0; i < length; i++ {
				items = append(items, Item{
					Label: fisher[i].Label,
					Rank:  a.X[i],
				})
			}
			return true
		})
		sort.Slice(items, func(i, j int) bool {
			return cmplx.Abs(items[i].Rank) < cmplx.Abs(items[j].Rank)
		})
		for _, item := range items {
			fmt.Printf("%.7f %.7f %s\n", cmplx.Abs(item.Rank), cmplx.Phase(item.Rank), item.Label)
		}
	}
	rank()

	correct := 0
	l2(func(a *tc128.V) bool {
		others := tf32.NewSet()
		others.Add("inputs", width, length)
		inputs := others.ByName["inputs"]
		for _, value := range a.X {
			inputs.X = append(inputs.X, float32(cmplx.Abs(value)))
		}
		others.Add("targets", 3, length)
		targets := others.ByName["targets"]
		targets.X = targets.X[:cap(targets.X)]
		for i := 0; i < length; i++ {
			targets.X[i*3+iris.Labels[fisher[i].Label]] = 1
		}

		set := tf32.NewSet()
		set.Add("weights", width, 3)
		weights := set.ByName["weights"]
		factor := math.Sqrt(2.0 / float64(weights.S[0]))
		for i := 0; i < cap(weights.X); i++ {
			weights.X = append(weights.X, float32(rnd.NormFloat64()*factor))
		}
		weights.States = make([][]float32, StateTotal)
		for i := range weights.States {
			weights.States[i] = make([]float32, len(weights.X))
		}
		set.Add("bias", 3, 1)
		bias := set.ByName["bias"]
		bias.X = bias.X[:cap(bias.X)]
		bias.States = make([][]float32, StateTotal)
		for i := range bias.States {
			bias.States[i] = make([]float32, len(bias.X))
		}

		softmax := tf32.U(occam.SphericalSoftmax)
		l1 := softmax(tf32.Add(tf32.Mul(set.Get("weights"), others.Get("inputs")), set.Get("bias")))
		cost := tf32.Avg(tf32.CrossEntropy(l1, others.Get("targets")))

		i := 1
		pow := func(x float32) float32 {
			return float32(math.Pow(float64(x), float64(i)))
		}

		points := make(plotter.XYs, 0, 8)

		// The stochastic gradient descent loop
		for i < 8*1024 {
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
					set.Weights[j].X[k] -= Eta * mhat / float32(math.Sqrt(float64(vhat))+1e-8)
				}
			}

			// Housekeeping
			end := time.Since(start)
			fmt.Println(i, total, end)
			set.Zero()
			others.Zero()

			if math.IsNaN(float64(total)) {
				fmt.Println(total)
				break
			}

			points = append(points, plotter.XY{X: float64(i), Y: float64(total)})
			i++
		}

		fmt.Println("----------------------------------------------------------")
		l1(func(a *tf32.V) bool {
			for i := 0; i < length; i++ {
				x, index, max := a.X[i*3:i*3+3], 0, float32(0.0)
				for i, value := range x {
					if value > max {
						index, max = i, value
					}
				}
				expected := iris.Labels[fisher[i].Label]
				fmt.Println(i, index, expected)
				if index == expected {
					correct++
				}
			}
			return true
		})
		fmt.Println("----------------------------------------------------------")

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

		err = p.Save(8*vg.Inch, 8*vg.Inch, "occam_top_complex_cost.png")
		if err != nil {
			panic(err)
		}
		return true
	})

	points := make(plotter.XYs, 0, 8)

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

		points = append(points, plotter.XY{X: float64(i), Y: float64(cmplx.Abs(total))})
		i++
	}

	rank()

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

	err = p.Save(8*vg.Inch, 8*vg.Inch, "occam_complex_cost.png")
	if err != nil {
		panic(err)
	}

	set.Save("occam_complex_set.w", 0, 0)

	fmt.Println("correct", correct, float64(correct)/150)
}
