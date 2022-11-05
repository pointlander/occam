// Copyright 2022 The Occam Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/occam"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

var (
	// FlagRND mixes random information into the rnn
	FlagRND = flag.Bool("rnd", false, "rnd mode")
)

// Source is a true random number source
type Source struct {
	Reader io.Reader
}

// NewSource creates a new true random source
func NewSource() *Source {
	in, err := os.Open("/dev/random")
	if err != nil {
		panic(err)
	}
	return &Source{
		Reader: in,
	}
}

// Int63 returns a true random int64
func (s *Source) Int63() int64 {
	return int64(s.Uint64() & ((1 << 63) - 1))
}

// Seed is a noop
func (s *Source) Seed(seed int64) {

}

// Uint64 returns a true random uint64
func (s *Source) Uint64() uint64 {
	data := make([]byte, 8)
	_, err := io.ReadFull(s.Reader, data)
	if err != nil {
		panic(err)
	}
	return binary.BigEndian.Uint64(data)
}

func main() {
	flag.Parse()

	rnda, rndb := rand.New(rand.NewSource(1)), rand.New(rand.NewSource(2))
	//rnda, rndb := rand.New(NewSource()), rand.New(NewSource())
	n := occam.NewNetwork(8, 8)
	state := make([]float64, 8)
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			if *FlagRND {
				n.Point.X[8*i+j] = 1
			} else {
				n.Point.X[8*i+j] = n.Rnd.Float32()
			}
		}
	}

	state[0] = 1
	for i := 0; i < 8*1024; i++ {
		total := n.Iterate(state)

		if math.IsNaN(float64(total)) {
			fmt.Println(total)
			break
		}

		n.L2(func(a *tf32.V) bool {
			for i, value := range a.X {
				state[i] = float64(value)
			}
			return true
		})
		fmt.Println(state)
		if *FlagRND {
			state[0] = rnda.Float64()
			state[1] = rndb.Float64()
		}
	}
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			fmt.Printf("%f ", n.Point.X[i*8+j])
		}
		fmt.Printf("\n")
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
}
