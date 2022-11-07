// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"time"

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

func main() {
	rnd := rand.New(rand.NewSource(1))

	vectors := make(map[string][]float32)
	reader, err := zip.OpenReader("wiki-news-300d-1M.vec.zip")
	if err != nil {
		panic(err)
	}
	defer reader.Close()

	for _, file := range reader.File {
		fmt.Printf("Contents of %s:\n", file.Name)
		file, err := file.Open()
		if err != nil {
			panic(err)
		}
		reader := bufio.NewReader(file)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					break
				}

				return
			}
			parts := strings.Split(line, " ")
			values := make([]float32, 0, len(parts)-1)
			for _, v := range parts[1:] {
				n, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
				if err != nil {
					panic(err)
				}
				values = append(values, float32(n))
			}
			max := float32(0)
			for _, v := range values {
				if v < 0 {
					v = -v
				}
				if v > max {
					max = v
				}
			}
			for i, v := range values {
				values[i] = v / max
			}
			vectors[strings.ToLower(strings.TrimSpace(parts[0]))] = values
		}
		file.Close()
	}

	data, err := ioutil.ReadFile("europarl-v7.de-en.en")
	if err != nil {
		panic(err)
	}
	en := strings.Split(string(data), "\n")
	english, length := make([][]string, 0, len(en)), 0
	for _, value := range en {
		value = strings.TrimSpace(value)
		if value != "" {
			parts := strings.Split(value, " ")
			for i, value := range parts {
				parts[i] = strings.ToLower(value)
			}
			if len(parts) > length {
				length = len(parts)
			}
			english = append(english, parts)
		}
	}

	width := 256

	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}

	others := tf32.NewSet()
	others.Add("symbols", 300, 1)
	symbols := others.ByName["symbols"]
	symbols.X = symbols.X[:cap(symbols.X)]

	// Create the weight data matrix
	set := tf32.NewSet()
	set.Add("points", width+300, 1024)
	set.Add("positions", width, length)
	for _, w := range set.Weights {
		//factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32((2*rnd.Float64() - 1) /*factor*/))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	begin1, end1 := 0, 0
	options1 := map[string]interface{}{
		"begin": &begin1,
		"end":   &end1,
	}

	// The neural network is the attention model from attention is all you need
	softmax := tf32.U(Softmax)
	l1 := softmax(tf32.Mul(set.Get("points"), tf32.Concat(others.Get("symbols"), tf32.Slice(set.Get("positions"), options1))))
	l2 := softmax(tf32.Mul(tf32.T(set.Get("points")), l1))
	cost := tf32.Entropy(l2)

	points := make(plotter.XYs, 0, 8)
	min := float32(math.MaxFloat32)

	// The stochastic gradient descent loop
	for i < 64*1024 {
		// Randomly select and load the input
		for {
			index := rnd.Intn(length)
			line := english[index]
			position := rnd.Intn(len(line))
			begin1 = position * width
			end1 = begin1 + width
			if vector, ok := vectors[line[position]]; ok {
				for i := range symbols.X {
					symbols.X[i] = vector[i]
				}
				break
			}
		}

		start := time.Now()
		// Calculate the gradients
		total := tf32.Gradient(cost).X[0]
		if total < min {
			min = total
		}

		// Update the point weights with the partial derivatives using adam
		b1, b2 := pow(B1), pow(B2)
		for j, w := range set.Weights {
			if w.N == "positions" {
				for k, d := range w.D {
					if k < begin1 || k >= end1 {
						continue
					}
					g := d
					m := B1*w.States[StateM][k] + (1-B1)*g
					v := B2*w.States[StateV][k] + (1-B2)*g*g
					w.States[StateM][k] = m
					w.States[StateV][k] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					set.Weights[j].X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
				}
				continue
			}
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
		others.Zero()

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

	fmt.Println("min", min)
}
