package bench

import (
	"encoding/binary"
	"math/rand"
	"time"
	"unsafe"
)

const CacheLineSize = 64

// sink prevents dead code elimination
var sink uint64

// SequentialRead performs a sequential scan reading 8 bytes per cache line.
// Returns throughput in GB/s.
//
//go:noinline
func SequentialRead(buf []byte, iters int) (gbPerSec float64, elapsed time.Duration) {
	n := len(buf)
	if n < 8 {
		return 0, 0
	}

	start := time.Now()
	var acc uint64
	for iter := 0; iter < iters; iter++ {
		for off := 0; off+8 <= n; off += 8 {
			acc += *(*uint64)(unsafe.Pointer(&buf[off]))
		}
	}
	elapsed = time.Since(start)
	sink += acc

	totalBytes := float64(n) * float64(iters)
	gbPerSec = totalBytes / elapsed.Seconds() / 1e9
	return gbPerSec, elapsed
}

// RandomRead performs random 8-byte reads at cache-line-aligned offsets.
// Returns operations per second and average latency in nanoseconds.
//
//go:noinline
func RandomRead(buf []byte, numAccesses int) (opsPerSec float64, avgLatencyNs float64) {
	n := len(buf)
	numSlots := n / CacheLineSize
	if numSlots == 0 {
		return 0, 0
	}

	// Build a shuffled array of offsets to defeat hardware prefetchers
	offsets := make([]int, numAccesses)
	rng := rand.New(rand.NewSource(42))
	for i := range offsets {
		offsets[i] = rng.Intn(numSlots) * CacheLineSize
	}

	start := time.Now()
	var acc uint64
	for _, off := range offsets {
		acc += binary.LittleEndian.Uint64(buf[off : off+8])
	}
	elapsed := time.Since(start)
	sink += acc

	opsPerSec = float64(numAccesses) / elapsed.Seconds()
	avgLatencyNs = float64(elapsed.Nanoseconds()) / float64(numAccesses)
	return opsPerSec, avgLatencyNs
}
