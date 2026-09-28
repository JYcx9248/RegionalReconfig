package placement

import (
	"fmt"
	"sort"
)

// Past answers which nodes have owned a partition before; *History implements it. A policy
// may use it to send a partition back where its data still is; nil means no history.
type Past interface {
	HasOwned(p int, n NodeID) bool
}

// Policy decides partition -> node placements.
type Policy interface {
	Name() string
	// Initial places numPartitions partitions on nodes.
	Initial(numPartitions int, nodes []NodeID) (Table, error)
	// Repartition returns a placement owned by exactly `nodes`, and the moves from old. past
	// may be nil; policies that do not use it ignore it.
	Repartition(old Table, nodes []NodeID, past Past) (Table, Changes, error)
}

// EvenPolicy balances partition counts and moves as few partitions as possible. Ported from
// Koala's EvenPartitionPolicy (internal/keyby/partition/evenPartitionPolicy.go):
//  1. partition counts differ by at most one;
//  2. partitions only move from existing to new nodes (scale-out) or from removed to
//     remaining nodes (scale-in), never between nodes that stay;
//  3. new nodes take partitions from all donors round-robin rather than from one donor.
//
// Differences from Koala: errors are returned instead of log.Fatalf; the extra partition of
// an uneven split goes to the nodes that already own the most (Koala gives it to the first
// nodes in the list, which fails when those currently own fewer); an unchanged node set is a
// no-op instead of a fatal error. Balancing by bytes or load is open question U3.
type EvenPolicy struct{}

func (EvenPolicy) Name() string { return "even" }

func (EvenPolicy) Initial(numPartitions int, nodes []NodeID) (Table, error) {
	if err := checkNodes(numPartitions, nodes); err != nil {
		return Table{}, err
	}
	owners := make([]NodeID, 0, numPartitions)
	base, rem := numPartitions/len(nodes), numPartitions%len(nodes)
	for i, n := range nodes {
		size := base
		if i < rem {
			size++
		}
		for j := 0; j < size; j++ {
			owners = append(owners, n)
		}
	}
	return Table{Owners: owners}, nil
}

func (EvenPolicy) Repartition(old Table, nodes []NodeID, _ Past) (Table, Changes, error) {
	return repartitionEven(old, nodes, nil)
}

// ReversiblePolicy is EvenPolicy with one rule added: on scale-in a partition goes back to a
// node that has owned it before, whenever such a node is still here and under its target
// count. Scale-out is unchanged, so scaling out and back in restores the previous placement
// exactly (the controller removes the nodes that joined), and with it every cached PQ code:
// the returning partitions name vectors their owner already holds, so the scale-in moves no
// codes at all.
//
// Koala does not do this -- its state changes with every input, so a copy left behind is
// stale and returning a bucket home saves nothing. Measured on its policy (ours reproduces
// it): after 4 -> 8 -> 4 only the 32 partitions that never moved are back with their original
// owner; of the 32 that moved, none is. See internal/placement/placement_test.go.
type ReversiblePolicy struct{}

func (ReversiblePolicy) Name() string { return "even-reversible" }

func (ReversiblePolicy) Initial(numPartitions int, nodes []NodeID) (Table, error) {
	return EvenPolicy{}.Initial(numPartitions, nodes)
}

func (ReversiblePolicy) Repartition(old Table, nodes []NodeID, past Past) (Table, Changes, error) {
	return repartitionEven(old, nodes, past)
}

// repartitionEven is Koala's even repartitioning; past != nil adds the scale-in rule above.
func repartitionEven(old Table, nodes []NodeID, past Past) (Table, Changes, error) {
	P := old.NumPartitions()
	if err := checkNodes(P, nodes); err != nil {
		return Table{}, nil, err
	}
	existing := old.Nodes() // first-appearance order (deterministic)
	owned := map[NodeID][]int{}
	for p, o := range old.Owners {
		owned[o] = append(owned[o], p)
	}
	want := map[NodeID]bool{}
	for _, n := range nodes {
		want[n] = true
	}
	var added, removed []NodeID
	for _, n := range nodes {
		if _, ok := owned[n]; !ok {
			added = append(added, n)
		}
	}
	for _, n := range existing {
		if !want[n] {
			removed = append(removed, n)
		}
	}
	next := old.Clone()
	switch {
	case len(added) == 0 && len(removed) == 0:
		return next, Changes{}, nil // same node set: nothing to do (rebalancing is U3)
	case len(added) > 0 && len(removed) > 0:
		return Table{}, nil, fmt.Errorf("placement: adding %v and removing %v at once is not "+
			"supported; scale out and in as two reconfigurations", added, removed)
	}

	// Target counts. The extra partitions of an uneven split go to nodes that own the most.
	order := append([]NodeID(nil), nodes...)
	sort.SliceStable(order, func(i, j int) bool { return len(owned[order[i]]) > len(owned[order[j]]) })
	desired := map[NodeID]int{}
	base, rem := P/len(nodes), P%len(nodes)
	for i, n := range order {
		desired[n] = base
		if i < rem {
			desired[n]++
		}
	}

	ch := Changes{}
	move := func(p int, from, to NodeID) {
		next.Owners[p] = to
		if ch[from] == nil {
			ch[from] = map[NodeID][]int{}
		}
		ch[from][to] = append(ch[from][to], p)
	}

	if len(added) > 0 {
		// Scale-out: existing nodes donate their excess (from the end of their lists).
		donate := map[NodeID][]int{}
		total := 0
		for _, n := range existing {
			excess := len(owned[n]) - desired[n]
			if excess < 0 {
				return Table{}, nil, fmt.Errorf("placement: node %d owns %d partitions, below "+
					"its target %d; rebalance first (U3)", n, len(owned[n]), desired[n])
			}
			donate[n] = owned[n][len(owned[n])-excess:]
			total += excess
		}
		di := 0
		for _, n := range added {
			need := desired[n]
			if need == 0 || need > total {
				return Table{}, nil, fmt.Errorf("placement: cannot give node %d %d partitions "+
					"(%d available)", n, need, total)
			}
			for need > 0 { // one partition per donor per pass (round-robin)
				d := existing[di]
				di = (di + 1) % len(existing)
				l := donate[d]
				if len(l) == 0 {
					continue
				}
				p := l[len(l)-1]
				donate[d] = l[:len(l)-1]
				move(p, d, n)
				need--
				total--
			}
		}
	} else {
		// Scale-in: removed nodes donate everything to the remaining nodes.
		donate := map[NodeID][]int{}
		supply := 0
		for _, n := range removed {
			donate[n] = append([]int(nil), owned[n]...)
			supply += len(owned[n])
		}
		demand := 0
		need := map[NodeID]int{}
		for _, n := range nodes {
			if desired[n] < len(owned[n]) {
				return Table{}, nil, fmt.Errorf("placement: node %d owns %d partitions, above "+
					"its target %d; rebalance first (U3)", n, len(owned[n]), desired[n])
			}
			need[n] = desired[n] - len(owned[n])
			demand += need[n]
		}
		if demand != supply {
			return Table{}, nil, fmt.Errorf("placement: scale-in supply %d != demand %d", supply, demand)
		}
		if past != nil { // first pass: home, while its target leaves room
			for _, d := range removed {
				keep := donate[d][:0]
				for _, p := range donate[d] {
					home, found := NodeID(0), false
					for _, n := range nodes {
						if need[n] > 0 && past.HasOwned(p, n) {
							home, found = n, true
							break
						}
					}
					if !found {
						keep = append(keep, p)
						continue
					}
					move(p, d, home)
					need[home]--
				}
				donate[d] = keep
			}
		}
		di := 0 // the rest: round-robin over the donors, as Koala does
		for _, n := range nodes {
			for need[n] > 0 {
				d := removed[di]
				di = (di + 1) % len(removed)
				l := donate[d]
				if len(l) == 0 {
					continue
				}
				p := l[len(l)-1]
				donate[d] = l[:len(l)-1]
				move(p, d, n)
				need[n]--
			}
		}
	}
	for _, m := range ch {
		for _, l := range m {
			sort.Ints(l)
		}
	}
	return next, ch, nil
}

func checkNodes(numPartitions int, nodes []NodeID) error {
	if len(nodes) == 0 {
		return fmt.Errorf("placement: no nodes")
	}
	if numPartitions < len(nodes) {
		return fmt.Errorf("placement: %d partitions cannot cover %d nodes", numPartitions, len(nodes))
	}
	seen := map[NodeID]bool{}
	for _, n := range nodes {
		if seen[n] {
			return fmt.Errorf("placement: node %d listed twice", n)
		}
		seen[n] = true
	}
	return nil
}

// NewPolicy returns the policy with the given name.
func NewPolicy(name string) (Policy, error) {
	switch name {
	case "", "even-reversible": // the decided design (U3)
		return ReversiblePolicy{}, nil
	case "even": // Koala's policy unchanged, kept as the baseline to compare against
		return EvenPolicy{}, nil
	case "weighted":
		return WeightedPolicy{}, nil
	}
	return nil, fmt.Errorf("placement: unknown policy %q", name)
}
