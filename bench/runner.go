package bench

import (
	"fmt"
	"numa-bench/numa"
	"runtime"
	"runtime/debug"
	"sync"
	"time"
)

const (
	DefaultSizeMB     = 256
	DefaultIters      = 10
	DefaultRandAccess = 2_000_000
	NumTrials         = 3
)

type Result struct {
	Scenario    string
	Workload    string
	AllocNode   int
	AccessNode  int
	GBPerSec    float64
	AvgLatNs    float64
	Duration    time.Duration
	GOMAXPROCS  int
}

// RunLocalRemote runs local and remote NUMA benchmarks.
func RunLocalRemote(topo *numa.Topology, sizeMB, iters, randAccesses int) []Result {
	size := sizeMB * 1024 * 1024
	var results []Result

	// Pick node 0 for the primary tests
	localNode := topo.Nodes[0]
	localCPU := localNode.CPUs[0]

	// Find the most distant remote node
	remoteNode := topo.Nodes[len(topo.Nodes)-1]
	maxDist := 0
	for _, n := range topo.Nodes {
		if n.ID == localNode.ID {
			continue
		}
		if localNode.Distances[n.ID] > maxDist {
			maxDist = localNode.Distances[n.ID]
			remoteNode = n
		}
	}
	remoteCPU := remoteNode.CPUs[0]

	// Local NUMA: allocate on node 0, access from node 0
	fmt.Printf("  Running local NUMA (node %d -> node %d)...\n", localNode.ID, localNode.ID)
	results = append(results, runPinned("local-numa", size, iters, randAccesses,
		localNode.ID, localCPU, localNode.ID)...)

	// Remote NUMA: allocate on node 0, access from remote node
	fmt.Printf("  Running remote NUMA (node %d -> node %d, distance=%d)...\n",
		localNode.ID, remoteNode.ID, maxDist)
	results = append(results, runPinned("remote-numa", size, iters, randAccesses,
		localNode.ID, remoteCPU, remoteNode.ID)...)

	return results
}

func runPinned(scenario string, size, iters, randAccesses, allocNode, accessCPU, accessNode int) []Result {
	var results []Result
	var mu sync.Mutex
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()

		// Pin to the alloc node's CPU for allocation
		allocNodeCPU := accessCPU
		// For local scenario, alloc and access are the same CPU.
		// For remote scenario, we need to pin to alloc node first,
		// then switch to access CPU. Find a CPU on the alloc node.
		if allocNode != accessNode {
			// We're already going to allocate via mbind, so the CPU
			// during allocation doesn't matter for placement.
		}
		_ = allocNodeCPU

		if err := numa.PinToCPU(accessCPU); err != nil {
			fmt.Printf("    warning: PinToCPU(%d) failed: %v\n", accessCPU, err)
		}

		// Allocate memory on the target node
		buf, usedMbind, err := numa.AllocOnNode(size, allocNode)
		if err != nil {
			fmt.Printf("    error: AllocOnNode failed: %v\n", err)
			return
		}
		defer numa.FreeNodeAlloc(buf)
		_ = usedMbind

		// Pin to the access CPU
		if err := numa.PinToCPU(accessCPU); err != nil {
			fmt.Printf("    warning: PinToCPU(%d) failed: %v\n", accessCPU, err)
		}

		// Warmup
		SequentialRead(buf, 1)
		RandomRead(buf, randAccesses/10)

		// Benchmark sequential read
		gbps, dur := SequentialRead(buf, iters)
		mu.Lock()
		results = append(results, Result{
			Scenario:   scenario,
			Workload:   "sequential-read",
			AllocNode:  allocNode,
			AccessNode: accessNode,
			GBPerSec:   gbps,
			Duration:   dur,
		})
		mu.Unlock()

		// Benchmark random read
		_, latNs := RandomRead(buf, randAccesses)
		mu.Lock()
		results = append(results, Result{
			Scenario:   scenario,
			Workload:   "random-read",
			AllocNode:  allocNode,
			AccessNode: accessNode,
			AvgLatNs:   latNs,
		})
		mu.Unlock()
	}()

	wg.Wait()
	return results
}

// RunGoDefault runs the Go-default scenario with no pinning or mbind.
func RunGoDefault(sizeMB, iters, randAccesses, gomaxprocs int) []Result {
	size := sizeMB * 1024 * 1024
	var results []Result

	for trial := 1; trial <= NumTrials; trial++ {
		fmt.Printf("  Go default trial %d/%d (GOMAXPROCS=%d)...\n", trial, NumTrials, gomaxprocs)
		r := runGoDefaultTrial(size, iters, randAccesses, gomaxprocs, trial)
		results = append(results, r...)
	}
	return results
}

func runGoDefaultTrial(size, iters, randAccesses, gomaxprocs, trial int) []Result {
	numGoroutines := gomaxprocs
	perSize := size

	type trialResult struct {
		seqGBps float64
		latNs   float64
	}

	resultsCh := make(chan trialResult, numGoroutines)
	var wg sync.WaitGroup

	// Barrier 1: wait group to ensure all gorutines finish warmup
	var warmupWg sync.WaitGroup
	warmupWg.Add(numGoroutines)

	// Barrier 2: channel to release all goroutines simultaneously for the benchmark
	startSeqBench := make(chan struct{})

	// Barrier 3: wait group to catch everyone finishing sequential read
	var seqWg sync.WaitGroup
	seqWg.Add(numGoroutines)

	// Barrier 4: channel to start Random Read benchmark for latency measurement
	startLatBench := make(chan struct{})

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			buf := make([]byte, perSize)
			// Touch all pages
			for i := 0; i < len(buf); i += 4096 {
				buf[i] = 1
			}

			// Yield to encourage scheduler migration
			for y := 0; y < 10; y++ {
				runtime.Gosched()
			}

			// Warmup
			SequentialRead(buf, 1)
			warmupWg.Done()

			<-startSeqBench
			// Benchmark
			SequentialRead(buf, iters)
			seqWg.Done()

			<-startLatBench
			_, latNs := RandomRead(buf, randAccesses/numGoroutines)

			resultsCh <- trialResult{latNs: latNs}
		}()
	}

	warmupWg.Wait() //Block until every worker goroutine has completed warmup

	t0 := time.Now()
	close(startSeqBench)      // Close channel to release all goroutines to do Sequential Read
	seqWg.Wait()              // wait until everyone finishes Sequentual Read
	elapsed := time.Since(t0)
	close(startLatBench)      // Close channel to release all goroutines to do Random Read

	wg.Wait()
	close(resultsCh)

	totalBytes := float64(numGoroutines) *
		(float64(perSize) / float64(CacheLineSize)) *
		8.0 *
		float64(iters)
	totalGBps := totalBytes / elapsed.Seconds() / 1e9

	var totalLat float64
	count := 0
	for r := range resultsCh {
		totalLat += r.latNs
		count++
	}

	return []Result{
		{
			Scenario:   fmt.Sprintf("go-default-trial-%d", trial),
			Workload:   "sequential-read",
			AllocNode:  -1,
			AccessNode: -1,
			GBPerSec:   totalGBps,
			GOMAXPROCS: gomaxprocs,
		},
		{
			Scenario:   fmt.Sprintf("go-default-trial-%d", trial),
			Workload:   "random-read",
			AllocNode:  -1,
			AccessNode: -1,
			AvgLatNs:   totalLat / float64(count),
			GOMAXPROCS: gomaxprocs,
		},
	}
}

// RunScalingSweep runs the Go-default scenario at every GOMAXPROCS value
// from 1 to NumCPU. Uses reduced iterations per step to keep total runtime
// reasonable.
func RunScalingSweep(topo *numa.Topology, sizeMB, iters, randAccesses int) []Result {
	totalCPUs := runtime.NumCPU()

	// Reduce iterations for the sweep since there are many steps
	sweepIters := iters / 2
	if sweepIters < 1 {
		sweepIters = 1
	}
	sweepRandAccesses := randAccesses / 2
	if sweepRandAccesses < 100000 {
		sweepRandAccesses = 100000
	}

	fmt.Printf("  Sweep: GOMAXPROCS 1..%d (iters=%d, rand-accesses=%d per step)\n",
		totalCPUs, sweepIters, sweepRandAccesses)

	var results []Result
	for procs := 1; procs <= totalCPUs; procs++ {
		fmt.Printf("  GOMAXPROCS=%d/%d\r", procs, totalCPUs)

		old := runtime.GOMAXPROCS(procs)

		trialResults := runGoDefaultTrial(sizeMB*1024*1024, sweepIters, sweepRandAccesses, procs, 1)
		for i := range trialResults {
			trialResults[i].Scenario = fmt.Sprintf("sweep-%d", procs)
			trialResults[i].GOMAXPROCS = procs
		}
		results = append(results, trialResults...)

		runtime.GOMAXPROCS(old)

		// Force GC between steps to reclaim goroutine buffers,
		// otherwise 192 steps * 4GB = OOM with GC disabled
		debug.SetGCPercent(100)
		runtime.GC()
		debug.SetGCPercent(-1)
	}
	fmt.Println() // clear the \r line

	return results
}
