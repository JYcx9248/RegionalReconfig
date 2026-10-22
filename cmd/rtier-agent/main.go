// rtier-agent: the per-node agent next to a running rtier_node (C++ data-node service).
//
//	rtier_node --index IDX --partitions PARTS --listen 127.0.0.1:7200 &
//	rtier-agent -config config.yaml -name n1 -node 127.0.0.1:7200 -work /tmp/rtier/n1
//
// config.yaml holds what every agent shares; the flags are this agent's own.
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
	path := flag.String("config", "", "the system configuration, config.yaml (default: every key at its default)")
	name := flag.String("name", "", "node name")
	node := flag.String("node", "", "rtier_node address")
	work := flag.String("work", "", "work directory")
	ctl := flag.String("controller", "", "override: controller address")
	qaddr := flag.String("query", "", "query listen address")
	baddr := flag.String("bulk", "", "bulk listen address")
	adv := flag.String("advertise", "", "host the other nodes reach this one at (default: the listen address's)")
	stats := flag.String("node-stats", "", "the data node's stats log (rtier_node --stats-file), forwarded to the controller")
	flag.Parse()

	cfg := config.DefaultAgent()
	if *path != "" {
		if err := config.Load(*path, &cfg); err != nil {
			log.Fatal(err)
		}
	}
	for dst, v := range map[*string]string{&cfg.Name: *name, &cfg.NodeAddr: *node, &cfg.WorkDir: *work,
		&cfg.Controller: *ctl, &cfg.QueryListen: *qaddr, &cfg.BulkListen: *baddr, &cfg.AdvertiseHost: *adv,
		&cfg.NodeStatsFile: *stats} {
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
