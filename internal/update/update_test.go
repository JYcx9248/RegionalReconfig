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

func TestMajorDue(t *testing.T) {
	lengths := map[uint32]uint32{1: 300, 2: 250, 3: 90, 4: 3, 5: 0, 6: 10, 7: 250}
	splits, merges := MajorDue(lengths, 200, 5, 0)
	if !reflect.DeepEqual(splits, []uint32{1, 2, 7}) || !reflect.DeepEqual(merges, []uint32{5, 4}) {
		t.Fatalf("got splits %v merges %v: want the longest first, then the shortest (an empty list too)", splits, merges)
	}
	splits, merges = MajorDue(lengths, 200, 5, 4)
	if !reflect.DeepEqual(splits, []uint32{1, 2, 7}) || !reflect.DeepEqual(merges, []uint32{5}) {
		t.Fatalf("at most 4: got splits %v merges %v", splits, merges)
	}
	if splits, merges = MajorDue(lengths, 0, 0, 0); splits != nil || merges != nil {
		t.Fatalf("thresholds off: got splits %v merges %v", splits, merges)
	}
}

func TestTakeLists(t *testing.T) {
	a, err := NewAllocator(0, 10, 0, 10)
	if err != nil {
		t.Fatal(err)
	}
	if got := a.TakeLists(2, 800); !reflect.DeepEqual(got, []uint32{800, 801}) {
		t.Fatalf("first lists: %v", got)
	}
	// Never handed out twice, even when the floor (the lists that exist) is below.
	if got := a.TakeLists(1, 800); !reflect.DeepEqual(got, []uint32{802}) || first(a.NextList()) != 803 {
		t.Fatalf("second take: %v, next %d", got, first(a.NextList()))
	}
	if got := a.TakeLists(1, 900); !reflect.DeepEqual(got, []uint32{900}) {
		t.Fatalf("a floor above: %v", got)
	}
}

func first(next uint64, _ uint32) uint64 { return next }
