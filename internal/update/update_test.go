package update

import (
	"reflect"
	"strings"
	"testing"

	"rtier/internal/epoch"
	"rtier/internal/partitioning"
	"rtier/internal/placement"
)

func TestAllocatorAppendsOnly(t *testing.T) {
	a, err := NewAllocator(100, 105, 640, 650)
	if err != nil {
		t.Fatal(err)
	}
	ids, locs, err := a.Take(3)
	if err != nil || !reflect.DeepEqual(ids, []uint32{100, 101, 102}) || !reflect.DeepEqual(locs, []uint32{640, 641, 642}) {
		t.Fatalf("first take: %v %v %v", ids, locs, err)
	}
	ids, _, err = a.Take(2)
	if err != nil || !reflect.DeepEqual(ids, []uint32{103, 104}) {
		t.Fatalf("second take: %v %v", ids, err)
	}
	// The ID space is full now; nothing is handed out twice, and a failed take reserves nothing.
	if _, _, err := a.Take(1); err == nil || !strings.Contains(err.Error(), "vector IDs") {
		t.Fatalf("take past the ID space: %v", err)
	}
	if id, loc := a.Next(); id != 105 || loc != 645 {
		t.Fatalf("next: %d %d", id, loc)
	}
	b, _ := NewAllocator(0, 10, 0, 2)
	if _, _, err := b.Take(3); err == nil || !strings.Contains(err.Error(), "locations") {
		t.Fatalf("take past the location space: %v", err)
	}
	if _, err := NewAllocator(5, 4, 0, 1); err == nil {
		t.Fatal("an ID start past the end was accepted")
	}
}

func TestByOwner(t *testing.T) {
	m := &partitioning.Manifest{NumLists: 6, NumPartitions: 3, ListPart: []uint32{0, 0, 1, 1, 2, 2}}
	tb := &epoch.Table{Epoch: 1, Placement: placement.Table{Owners: []placement.NodeID{7, 3, 7}}}
	ins := []Insert{
		{ID: 10, Lists: []uint32{0, 1, 4}}, // partitions 0, 0, 2: only node 7, once
		{ID: 11, Lists: []uint32{2, 5}},    // partitions 1, 2: nodes 3 and 7
		{ID: 12, Lists: []uint32{3}},       // partition 1: node 3
	}
	got := ByOwner(ins, m, tb)
	want := map[placement.NodeID][]int{7: {0, 1}, 3: {1, 2}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}
