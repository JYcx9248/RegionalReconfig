package metrics

import (
	"syscall"
	"unsafe"
)

// Monotonic is CLOCK_MONOTONIC in seconds: the clock of the experiment scripts (Python's
// time.monotonic(), clock.json) and of the data nodes' stats logs. The controller stamps what
// the nodes forward with it when it arrives, so that a run has one clock however many machines
// it spans (as Koala's metric collector stamps the batches it receives).
func Monotonic() float64 {
	var ts syscall.Timespec
	const clockMonotonic = 1
	syscall.Syscall(syscall.SYS_CLOCK_GETTIME, clockMonotonic, uintptr(unsafe.Pointer(&ts)), 0)
	return float64(ts.Sec) + float64(ts.Nsec)/1e9
}
