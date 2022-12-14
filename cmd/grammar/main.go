// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"compress/gzip"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
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

// PartOfSpeech is a part of speech with examples
var PartsOfSpeech = map[string][10]string{
	"noun":         {"city", "new york", "banana", "bananas", "family", "ice cream", "table", "anger"},
	"pronoun":      {"i", "he", "him", "you", "we", "him", "her", "yours", "theirs", "someone"},
	"verb":         {"swim", "realize", "run", "walk", "laugh", "have", "promise", "invite", "listen", "running"},
	"adjective":    {"adventurous", "beautiful", "calm", "dark", "drab", "easy", "foolish", "grieving", "helpful", "important"},
	"adverb":       {"quickly", "happily", "easily", "slowly", "angrily", "honestly", "carefully", "quietly", "suddenly", "safely"},
	"preposition":  {"in", "on", "at", "near", "for", "with", "about", "between", "under", "behind"},
	"conjunction":  {"and", "or", "but", "because", "so", "yet", "although", "since", "unless", "if"},
	"interjection": {"ouch", "wow", "oh", "ah", "oops", "yikes", "yay", "yuck", "yippee", "yup"},
}

// Vector is a word vector
type Vector struct {
	Word   string
	Vector []float32
}

// Vectors is a set of word vectors
type Vectors struct {
	List       []Vector
	Dictionary map[string]Vector
}

// NewVectors creates a new word vector set
func NewVectors(file string) Vectors {
	vectors := Vectors{
		Dictionary: make(map[string]Vector),
	}
	in, err := os.Open(file)
	if err != nil {
		panic(err)
	}
	defer in.Close()

	gzipReader, err := gzip.NewReader(in)
	if err != nil {
		panic(err)
	}
	reader := bufio.NewReader(gzipReader)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
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
		word := strings.ToLower(strings.TrimSpace(parts[0]))
		vector := Vector{
			Word:   word,
			Vector: values,
		}
		vectors.List = append(vectors.List, vector)
		vectors.Dictionary[word] = vector
		if len(vector.Vector) == 0 {
			fmt.Println(vector)
		}
	}
	return vectors
}

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

	env := NewVectors("cc.en.300.vec.gz")
	dev := NewVectors("cc.de.300.vec.gz")

	width := 300

	if *FlagInfer != "" {
		others := tf32.NewSet()
		others.Add("symbols", width, 1)
		symbols := others.ByName["symbols"]
		symbols.X = symbols.X[:cap(symbols.X)]

		// Create the weight data matrix
		set := tf32.NewSet()
		set.Open(*FlagInfer)
		points := set.ByName["points"]

		softmax := tf32.U(Softmax)
		l1 := softmax(tf32.Mul(set.Get("points"), others.Get("symbols")))

		type Point struct {
			Index int
			Rank  float32
		}
		type Input struct {
			Points []Point
			Label  string
		}

		cluster := func(word string, vectors Vectors) Input {
			vector := vectors.Dictionary[word]
			for i, measure := range vector.Vector {
				symbols.X[i] = float32(measure)
			}
			// Calculate the l1 output of the neural network
			var input Input
			l1(func(a *tf32.V) bool {
				points := make([]Point, 0, width)
				for j, value := range a.X {
					points = append(points, Point{
						Index: j,
						Rank:  value,
					})
				}
				sort.Slice(points, func(i, j int) bool {
					return points[i].Rank > points[j].Rank
				})
				input = Input{
					Points: points,
					Label:  word,
				}
				return true
			})
			return input
		}
		a := cluster("car", env)
		b := cluster("truck", env)
		for _, value := range a.Points {
			if value.Rank == 0 {
				continue
			}
			fmt.Printf("%d %f ", value.Index, value.Rank)
		}
		fmt.Printf("\n")
		for _, value := range b.Points {
			if value.Rank == 0 {
				continue
			}
			fmt.Printf("%d %f ", value.Index, value.Rank)
		}
		fmt.Printf("\n")

		for part, words := range PartsOfSpeech {
			common := make(map[int]int)
			for _, word := range words {
				fmt.Printf("%12s ", part)
				value := cluster(word, env)
				for _, point := range value.Points[:20] {
					common[point.Index]++
					fmt.Printf("%4d ", point.Index)
				}
				fmt.Printf("\n")
			}
			maxIndex, max := 0, 0
			for index, count := range common {
				if count > max {
					max, maxIndex = count, index
				}
			}
			fmt.Println(maxIndex)
		}

		l1 = tf32.Mul(set.Get("points"), others.Get("symbols"))

		g, y := image.NewGray16(image.Rect(0, 0, 1024, 1024)), 0
		for i := 0; i < width*1024; i += width {
			copy(symbols.X, points.X[i:i+width])
			max, min := float32(0), float32(math.MaxFloat32)
			l1(func(a *tf32.V) bool {
				for _, value := range a.X {
					if value > max {
						max = value
					}
					if value < min {
						min = value
					}
				}
				return true
			})
			l1(func(a *tf32.V) bool {
				for i, value := range a.X {
					g.SetGray16(i, y, color.Gray16{uint16(65535 * (value - min) / (max - min))})
				}
				return true
			})
			y++
		}

		f, err := os.Create("output.png")
		if err != nil {
			panic(err)
		}
		defer f.Close()
		err = png.Encode(f, g)
		if err != nil {
			panic(err)
		}
		return
	}
	/*data, err := ioutil.ReadFile("europarl-v7.de-en.en")
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
	}*/

	i := 1
	pow := func(x float32) float32 {
		y := math.Pow(float64(x), float64(i))
		if math.IsNaN(y) || math.IsInf(y, 0) {
			return 0
		}
		return float32(y)
	}

	others := tf32.NewSet()
	others.Add("symbols", width, 1)
	symbols := others.ByName["symbols"]
	symbols.X = symbols.X[:cap(symbols.X)]

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
	l1 := softmax(tf32.Mul(set.Get("points"), others.Get("symbols")))
	l2 := softmax(tf32.Mul(tf32.T(set.Get("points")), l1))
	cost := tf32.Entropy(l2)

	points := make(plotter.XYs, 0, 8)
	min := float32(math.MaxFloat32)

	// The stochastic gradient descent loop
	for i < 256*1024 {
		// Randomly select and load the input
		vectors := env
		if *FlagTrain == "de" {
			vectors = dev
		}
		/*if i < 128*1024 {
			vectors = env
		} else if i < 256*1024 {
			vectors = dev
		} else {
			vectors = env
			if rnd.Intn(2) == 0 {
				vectors = dev
			}
		}*/
		vector := vectors.List[rnd.Intn(len(vectors.List))]
		for i := range symbols.X {
			symbols.X[i] = vector.Vector[i]
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

	set.Save(fmt.Sprintf("%s_set.w", *FlagTrain), 0, 0)

	fmt.Println("min", min)
}
