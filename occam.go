// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package occam

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
	"gonum.org/v1/plot/plotter"
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

// Network is a clustering neural network
type Network struct {
	Rnd    *rand.Rand
	Width  int
	Length int
	Set    tf32.Set
	Others tf32.Set
	Input  *tf32.V
	Point  *tf32.V
	L1     tf32.Meta
	L2     tf32.Meta
	Cost   tf32.Meta
	I      int
	Points plotter.XYs
}

func (n *Network) pow(x float32) float32 {
	y := math.Pow(float64(x), float64(n.I))
	if math.IsNaN(y) || math.IsInf(y, 0) {
		return 0
	}
	return float32(y)
}

// Creates a new neural network
func NewNetwork(width, length int) *Network {
	n := Network{
		Rnd:    rand.New(rand.NewSource(1)),
		Width:  width,
		Length: length,
		I:      1,
	}

	// Create the input data matrix
	n.Others = tf32.NewSet()
	n.Others.Add("input", width, 1)
	n.Input = n.Others.ByName["input"]
	n.Input.X = n.Input.X[:cap(n.Input.X)]

	// Create the weight data matrix
	n.Set = tf32.NewSet()
	n.Set.Add("points", width, length)
	n.Point = n.Set.ByName["points"]
	n.Point.X = n.Point.X[:cap(n.Point.X)]
	n.Point.States = make([][]float32, StateTotal)
	for i := range n.Point.States {
		n.Point.States[i] = make([]float32, len(n.Point.X))
	}

	// The neural network is the attention model from attention is all you need
	softmax := tf32.U(Softmax)
	n.L1 = softmax(tf32.Mul(n.Set.Get("points"), n.Others.Get("input")))
	n.L2 = softmax(tf32.Mul(tf32.T(n.Set.Get("points")), n.L1))
	n.Cost = tf32.Entropy(n.L2)

	n.Points = make(plotter.XYs, 0, 8)

	return &n
}

// Iterate does a gradient descent operation
func (n *Network) Iterate(data []float64) float32 {
	for i, measure := range data {
		n.Input.X[i] = float32(measure)
	}

	start := time.Now()
	// Calculate the gradients
	total := tf32.Gradient(n.Cost).X[0]

	// Update the point weights with the partial derivatives using adam
	b1, b2 := n.pow(B1), n.pow(B2)
	for j, w := range n.Set.Weights {
		for k, d := range w.D {
			g := d
			m := B1*w.States[StateM][k] + (1-B1)*g
			v := B2*w.States[StateV][k] + (1-B2)*g*g
			w.States[StateM][k] = m
			w.States[StateV][k] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			n.Set.Weights[j].X[k] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
		}
	}

	// Housekeeping
	end := time.Since(start)
	fmt.Println(n.I, total, end)
	n.Set.Zero()
	n.Others.Zero()
	n.Points = append(n.Points, plotter.XY{X: float64(n.I), Y: float64(total)})
	n.I++

	return total
}

// Analyzer calculates properties of the network
func (n *Network) Analyzer(fisher []iris.Iris) {
	// For each input, label and sort the points in terms of distance to the input
	type Point struct {
		Index int
		Rank  float32
	}
	type Input struct {
		Points []Point
		Label  string
	}
	type Node struct {
		Nodes map[int]*Node
		Ranks []float32
		Label []string
	}
	var build func(input Input, depth int, node *Node)
	build = func(input Input, depth int, node *Node) {
		length := n.Length
		if depth >= length {
			return
		}
		if node.Nodes == nil {
			node.Nodes = make(map[int]*Node)
		}
		n := node.Nodes[input.Points[depth].Index]
		if n == nil {
			n = &Node{}
		}
		n.Ranks = append(n.Ranks, input.Points[depth].Rank)
		if depth == length-1 {
			n.Label = append(n.Label, input.Label)
		}
		node.Nodes[input.Points[depth].Index] = n
		build(input, depth+1, n)
	}
	inputs := make([]Input, 0, n.Length)
	for i := 0; i < n.Length; i++ {
		// Load the input
		sample := fisher[i]
		for i, measure := range sample.Measures {
			n.Input.X[i] = float32(measure)
		}
		// Calculate the l1 output of the neural network
		n.L1(func(a *tf32.V) bool {
			points := make([]Point, 0, n.Length)
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
			if index == n.Length {
				return false
			}
		}
		return inputs[i].Points[index].Index < inputs[j].Points[index].Index
	})

	node := &Node{}
	for _, input := range inputs {
		build(input, 0, node)
	}

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
	fmt.Println(same, n.Length, float64(same)/float64(n.Length))
}
