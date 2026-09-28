// rtier-agent: the per-node agent next to a running rtier_node (C++ data-node service).
//
//	rtier_node --index IDX --partitions PARTS --listen 127.0.0.1:7200 &
//	rtier-agent -config configs/agent.json -name n1 -node 127.0.0.1:7200 -work /tmp/rtier/n1
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"rtier/internal/agent"
	"rtier/internal/config"
)

func main() {
	path := flag.String("config", "", "JSON config (see configs/agent.example.json)")
	name := flag.String("name", "", "override: node name")
	node := flag.String("node", "", "override: rtier_node address")
	work := flag.String("work", "", "override: work directory")
	ctl := flag.String("controller", "", "override: controller address")
	qaddr := flag.String("query", "", "override: query listen address")
	baddr := flag.String("bulk", "", "override: bulk listen address")
	flag.Parse()

	cfg := config.DefaultAgent()
	if *path != "" {
		if err := config.Load(*path, &cfg); err != nil {
			log.Fatal(err)
		}
	}
	for dst, v := range map[*string]string{&cfg.Name: *name, &cfg.NodeAddr: *node, &cfg.WorkDir: *work,
		&cfg.Controller: *ctl, &cfg.QueryListen: *qaddr, &cfg.BulkListen: *baddr} {
		if v != "" {
			*dst = v
		}
	}
	a, err := agent.New(cfg, agent.Options{})
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := a.Run(ctx); err != nil {
		log.Fatal(err)
	}
}
