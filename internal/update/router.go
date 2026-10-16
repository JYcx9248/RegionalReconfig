package update

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"
)

// StartRouter starts the update controller's router: an rtier_node (bin) on the index with its
// navigation graph and codebook, no partition and a PQ budget of one code -- all that ASSIGN
// needs, and no raw vector (the index may lack its page file). Its log goes to logw. stop
// interrupts it, waits, and returns how it exited.
func StartRouter(bin, index, parts string, logw io.Writer) (addr string, stop func() error, err error) {
	cmd := exec.Command(bin, "--index", index, "--partitions", parts, "--listen", "127.0.0.1:0",
		"--graph", "index", "--pq-capacity", "1", "--workers", "1")
	cmd.Stderr = logw
	out, err := cmd.StdoutPipe()
	if err != nil {
		return "", nil, err
	}
	if err := cmd.Start(); err != nil {
		return "", nil, fmt.Errorf("update: router %s: %w", bin, err)
	}
	stop = func() error {
		cmd.Process.Signal(os.Interrupt)
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		select {
		case err := <-done:
			return err
		case <-time.After(10 * time.Second):
			cmd.Process.Kill()
			<-done
			return fmt.Errorf("update: router %s did not stop within 10 s", bin)
		}
	}
	br := bufio.NewReader(out)
	line, err := br.ReadString('\n')
	if err != nil || !strings.HasPrefix(line, "READY tcp=") {
		stop()
		return "", nil, fmt.Errorf("update: router %s did not start: %q %v", bin, line, err)
	}
	go io.Copy(io.Discard, br) // whatever else it prints must not fill the pipe
	return "127.0.0.1:" + strings.TrimSpace(strings.TrimPrefix(line, "READY tcp=")), stop, nil
}
