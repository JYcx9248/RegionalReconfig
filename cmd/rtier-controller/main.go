// rtier-controller: the regional tier's control plane.
//
//	rtier-controller -config config.yaml
//	rtier-controller -defaults > config.yaml     # every key at its default
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"rtier/internal/config"
	"rtier/internal/controller"
)

func main() {
	path := flag.String("config", "", "the system configuration, config.yaml (default: every key at its default)")
	defaults := flag.Bool("defaults", false, "print config.yaml with every key at its default, the agents' too, and exit")
	listen := flag.String("listen", "", "override: control-plane address")
	api := flag.String("api", "", "override: API address")
	initial := flag.Int("initial-nodes", 0, "override: data nodes of the first epoch")
	protocol := flag.String("protocol", "", "override: lazy | lazy-stream | copy-then-flip | stop-and-copy")
	flag.Parse()
	if *defaults {
		b, err := config.Defaults()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(string(b))
		return
	}

	cfg := config.DefaultController()
	if *path != "" {
		if err := config.Load(*path, &cfg); err != nil {
			log.Fatal(err)
		}
	}
	if *listen != "" {
		cfg.Listen = *listen
	}
	if *api != "" {
		cfg.APIListen = *api
	}
	if *initial > 0 {
		cfg.InitialNodes = *initial
	}
	if *protocol != "" {
		cfg.Protocol = *protocol
	}
	c, err := controller.New(cfg)
	if err != nil {
		log.Fatal(err)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if err := c.Run(ctx); err != nil {
		log.Fatal(err)
	}
}
