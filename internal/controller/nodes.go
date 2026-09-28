package controller

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"rtier/internal/ctrl"
	"rtier/internal/epoch"
	"rtier/internal/placement"
	"rtier/internal/protocol"
)

// ManagedNode is one registered agent (Koala's ManagedWorker): its addresses and the control
// connection the controller sends commands on.
type ManagedNode struct {
	ID   placement.NodeID
	Name string
	Info epoch.NodeInfo
	Conn *ctrl.Conn
}

// Alive reports whether the control connection is up.
func (n *ManagedNode) Alive() bool {
	select {
	case <-n.Conn.Done():
		return false
	default:
		return true
	}
}

// call sends one command and waits for the reply, with a timeout.
func (n *ManagedNode) call(ctx context.Context, timeout time.Duration, method string, req, resp any) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	if err := n.Conn.Call(ctx, method, req, resp); err != nil {
		return &NodeError{Node: n.ID, Name: n.Name, Method: method, Err: err}
	}
	return nil
}

// NodeError wraps a failed command.
type NodeError struct {
	Node   placement.NodeID
	Name   string
	Method string
	Err    error
}

func (e *NodeError) Error() string {
	return fmt.Sprintf("node %s (%d) %s: %v", e.Name, e.Node, e.Method, e.Err)
}
func (e *NodeError) Unwrap() error { return e.Err }

// NodeManager tracks registered nodes (Koala's WorkerManager: monotonic IDs, the pool of
// available workers).
type NodeManager struct {
	mu     sync.Mutex
	nodes  map[placement.NodeID]*ManagedNode
	nextID placement.NodeID
	added  chan struct{} // signaled on every registration
}

func NewNodeManager() *NodeManager {
	return &NodeManager{nodes: map[placement.NodeID]*ManagedNode{}, nextID: 1, added: make(chan struct{}, 1)}
}

// Add registers a node and assigns its ID.
func (m *NodeManager) Add(name string, info epoch.NodeInfo, conn *ctrl.Conn) *ManagedNode {
	m.mu.Lock()
	id := m.nextID
	m.nextID++
	info.ID = id
	info.Name = name
	n := &ManagedNode{ID: id, Name: name, Info: info, Conn: conn}
	m.nodes[id] = n
	m.mu.Unlock()
	select {
	case m.added <- struct{}{}:
	default:
	}
	return n
}

// Get returns node id.
func (m *NodeManager) Get(id placement.NodeID) (*ManagedNode, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	n, ok := m.nodes[id]
	return n, ok
}

// All returns every registered node, by ID.
func (m *NodeManager) All() []*ManagedNode {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*ManagedNode, 0, len(m.nodes))
	for _, n := range m.nodes {
		out = append(out, n)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Idle returns live nodes that are not part of table t (the pool scale-out draws from), by ID.
func (m *NodeManager) Idle(t *epoch.Table) []*ManagedNode {
	var out []*ManagedNode
	for _, n := range m.All() {
		if !n.Alive() {
			continue
		}
		if t != nil {
			if _, in := t.Nodes[n.ID]; in {
				continue
			}
		}
		out = append(out, n)
	}
	return out
}

// WaitFor blocks until at least k live nodes are registered.
func (m *NodeManager) WaitFor(ctx context.Context, k int) ([]*ManagedNode, error) {
	for {
		var live []*ManagedNode
		for _, n := range m.All() {
			if n.Alive() {
				live = append(live, n)
			}
		}
		if len(live) >= k {
			return live[:k], nil
		}
		select {
		case <-m.added:
		case <-time.After(time.Second):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// Summaries for Status.
func (m *NodeManager) Summaries(t *epoch.Table) []protocol.NodeSummary {
	var out []protocol.NodeSummary
	for _, n := range m.All() {
		idle := true
		if t != nil {
			_, in := t.Nodes[n.ID]
			idle = !in
		}
		out = append(out, protocol.NodeSummary{ID: n.ID, Name: n.Name, Info: n.Info, Idle: idle, Alive: n.Alive()})
	}
	return out
}
