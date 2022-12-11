// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package occam

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/pointlander/datum/iris"
	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/levenshtein"
	"github.com/pointlander/pagerank"
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

// SphericalSoftmax is the spherical softmax function
// https://arxiv.org/abs/1511.05042
func SphericalSoftmax(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
	const E = .0
	c, size, width := tf32.NewV(a.S...), len(a.X), a.S[0]
	values, sums, row := make([]float32, width), make([]float32, a.S[1]), 0
	for i := 0; i < size; i += width {
		sum := float32(0.0)
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
	_ = softmax
	spherical := tf32.U(SphericalSoftmax)
	_ = spherical
	n.L1 = softmax(tf32.Mul(n.Set.Get("points"), n.Others.Get("input")))
	n.L2 = softmax(tf32.T(tf32.Mul(n.L1, tf32.T(n.Set.Get("points")))))
	n.Cost = tf32.Entropy(n.L2)

	n.Points = make(plotter.XYs, 0, 8)

	return &n
}

// Entropy is the self entropy of a point
type Entropy struct {
	Entropy  float32
	Label    string
	Measures []float64
}

// GetEntropy returns the entropy of the network
func (n *Network) GetEntropy(inputs []iris.Iris) []Entropy {
	outputs := make([]Entropy, 0, len(inputs))
	for i := 0; i < len(inputs); i++ {
		// Load the input
		sample := inputs[i]
		for i, measure := range sample.Measures {
			n.Input.X[i] = float32(measure)
		}
		// Calculate the l1 output of the neural network
		n.Cost(func(a *tf32.V) bool {
			outputs = append(outputs, Entropy{
				Entropy:  a.X[0],
				Label:    sample.Label,
				Measures: sample.Measures,
			})
			return true
		})
	}
	return outputs
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

func (n *Network) GetVectors(inputs []iris.Iris) []iris.Iris {
	outputs := make([]iris.Iris, 0, len(inputs))
	for i := 0; i < n.Length; i++ {
		// Load the input
		sample := inputs[i]
		for i, measure := range sample.Measures {
			n.Input.X[i] = float32(measure)
		}
		// Calculate the l1 output of the neural network
		n.L1(func(a *tf32.V) bool {
			vectors := make([]float64, len(a.X))
			for i, x := range a.X {
				vectors[i] = float64(x)
			}
			outputs = append(outputs, iris.Iris{
				Measures: vectors,
				Label:    sample.Label,
			})
			return true
		})
	}
	return outputs
}

// Analyzer calculates properties of the network
func (n *Network) Analyzer(in []iris.Iris) {
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
	vectors := n.GetVectors(in)
	for i := 0; i < n.Length; i++ {
		vector := vectors[i]
		points := make([]Point, 0, n.Length)
		for j, value := range vector.Measures {
			points = append(points, Point{
				Index: j,
				Rank:  float32(value),
			})
		}
		sort.Slice(points, func(i, j int) bool {
			return points[i].Rank > points[j].Rank
		})
		inputs = append(inputs, Input{
			Points: points,
			Label:  vector.Label,
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
	var translate func(node *Node, tree *[]*opts.TreeData)
	tree := make([]*opts.TreeData, 0, 8)
	translate = func(node *Node, tree *[]*opts.TreeData) {
		if node == nil || tree == nil {
			return
		}
		if len(node.Nodes) == 1 {
			for i, n := range node.Nodes {
				if n.Nodes == nil {
					label := ""
					for _, l := range n.Label {
						label += fmt.Sprintf("%s-", l)
					}
					t := opts.TreeData{
						Name: fmt.Sprintf("%d-%s", i, label),
					}
					*tree = append(*tree, &t)
					break
				}
				translate(n, tree)
			}
			return
		}
		for i, n := range node.Nodes {
			t := opts.TreeData{
				Name:     fmt.Sprintf("%d", i),
				Children: make([]*opts.TreeData, 0, 8),
			}
			translate(n, &t.Children)
			*tree = append(*tree, &t)
		}
	}
	translate(node, &tree)
	t := []opts.TreeData{
		{
			Name:     "Root",
			Children: tree,
		},
	}
	graph := charts.NewTree()
	graph.SetGlobalOptions(
		charts.WithInitializationOpts(opts.Initialization{Width: "100%", Height: "500vh"}),
		charts.WithTitleOpts(opts.Title{Title: "basic tree example"}),
		charts.WithTooltipOpts(opts.Tooltip{Show: false}),
	)
	graph.AddSeries("tree", t).
		SetSeriesOptions(
			charts.WithTreeOpts(
				opts.TreeChart{
					Layout:           "orthogonal",
					Orient:           "LR",
					InitialTreeDepth: -1,
					Leaves: &opts.TreeLeaves{
						Label: &opts.Label{Show: true, Position: "right", Color: "Black"},
					},
				},
			),
			charts.WithLabelOpts(opts.Label{Show: true, Position: "top", Color: "Black"}),
		)
	page := components.NewPage()
	page.AddCharts(
		graph,
	)
	f, err := os.Create("tree.html")
	if err != nil {
		panic(err)

	}
	page.Render(io.MultiWriter(f))

	// Count how many inputs have the same label as their nearest neighbor
	same := 0
	for i, label := range inputs {
		min, index := math.MaxInt, 0
		for j, l := range inputs {
			if i == j {
				continue
			}
			a, b := make([]int, 0, 8), make([]int, 0, 8)
			for k, value := range label.Points {
				a = append(a, value.Index)
				b = append(b, l.Points[k].Index)
			}
			total := levenshtein.ComputeDistance(a, b)
			if total < min {
				min, index = total, j
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

	type Point64 struct {
		Index int
		Rank  float64
	}
	g := pagerank.NewGraph64()
	for i, vector := range vectors {
		for j, weight := range vector.Measures {
			g.Link(uint64(i), uint64(j), weight)
		}
	}
	ranks := make([]Point64, n.Length)
	g.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node].Rank = rank
		ranks[node].Index = int(node)
	})
	sort.Slice(ranks, func(i, j int) bool {
		return ranks[i].Rank > ranks[j].Rank
	})
	for _, rank := range ranks {
		fmt.Printf("%03d %.16f\n", rank.Index, rank.Rank)
	}
}
