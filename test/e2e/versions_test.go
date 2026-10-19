package e2e

import (
	"context"
	"path/filepath"
	"reflect"
	"testing"

	"rtier/internal/nodeclient"
	"rtier/internal/partitioning"
)

// TestPartitionVersions checks the versions of a partition's lists over the wire (U14): FILTER
// and RERANK say which version their query's epoch sees, the node refuses one it does not
// hold, a second version loads next to the first, and an eviction drops them all.
func TestPartitionVersions(t *testing.T) {
	f := getFixture(t)
	addr, stop, err := startNode(f, "--load", "all", "--graph", "index")
	if err != nil {
		t.Fatal(err)
	}
	defer stop()
	m, err := partitioning.LoadManifest(f.parts)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	nc := nodeclient.New(addr, 1)
	defer nc.Close()
	versionsOf := func(p int) []uint32 {
		in, err := nc.Info(ctx)
		if err != nil {
			t.Fatal(err)
		}
		var vs []uint32
		for _, pv := range in.Versions {
			if int(pv[0]) == p {
				vs = append(vs, pv[1])
			}
		}
		return vs
	}

	q := f.queries.Row(0)
	all, err := nc.Navigate(ctx, 1, nodeclient.NewestGraph, q, testNProbe, 0)
	if err != nil {
		t.Fatal(err)
	}
	p := m.PartitionOf(all[0])
	var lists []uint32 // the probed lists of partition p
	for _, c := range all {
		if m.PartitionOf(c) == p {
			lists = append(lists, c)
		}
	}
	if got := versionsOf(p); !reflect.DeepEqual(got, []uint32{0}) {
		t.Fatalf("after --load all, partition %d is at versions %v, want [0]", p, got)
	}
	versions := func(v uint32) []uint32 {
		vs := make([]uint32, len(lists))
		for i := range vs {
			vs[i] = v
		}
		return vs
	}
	at0, _, err := nc.Filter(ctx, 1, q, lists, versions(0), maxRerank)
	if err != nil || len(at0) == 0 {
		t.Fatalf("FILTER at version 0: %d candidates, %v", len(at0), err)
	}
	if _, _, err := nc.Filter(ctx, 1, q, lists, versions(1), maxRerank); !nodeclient.IsNotResident(err) {
		t.Fatalf("FILTER at version 1, which the node does not hold: got %v, want not resident", err)
	}

	// Version 1 next to version 0 (the same lists here): both serve.
	seg := filepath.Join(f.parts, partitioning.SegmentName(p))
	if err := nc.LoadPartition(ctx, p, 1, seg, false); err != nil {
		t.Fatal(err)
	}
	if got := versionsOf(p); !reflect.DeepEqual(got, []uint32{0, 1}) {
		t.Fatalf("after loading version 1: versions %v, want [0 1]", got)
	}
	at1, _, err := nc.Filter(ctx, 1, q, lists, versions(1), maxRerank)
	if err != nil || !reflect.DeepEqual(at1, at0) {
		t.Fatalf("FILTER at version 1: %v (%d candidates, %d at version 0)", err, len(at1), len(at0))
	}
	ids := make([]uint32, 0, 16)
	for _, c := range at1[:min(16, len(at1))] {
		ids = append(ids, c.ID)
	}
	if _, _, err := nc.Rerank(ctx, 1, q, ids, lists, versions(1), 10); err != nil {
		t.Fatalf("RERANK at version 1: %v", err)
	}

	// An eviction drops every version.
	if was, err := nc.EvictPartition(ctx, p); err != nil || !was {
		t.Fatalf("evict: %v %v", was, err)
	}
	if got := versionsOf(p); len(got) != 0 {
		t.Fatalf("after the eviction: versions %v", got)
	}
	if _, _, err := nc.Filter(ctx, 1, q, lists, versions(0), maxRerank); !nodeclient.IsNotResident(err) {
		t.Fatalf("FILTER after the eviction: got %v, want not resident", err)
	}
}
