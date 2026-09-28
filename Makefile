# rtier: build the C++ engine and the Go binaries, run the tests.
#
#   make              # engine (CPU) + Go binaries in bin/
#   make CUDA=ON      # engine with the GPU filter backend
#   make test         # engine unit tests + Go tests (race detector) + end-to-end tests

ENGINE_BUILD ?= engine/build
CUDA         ?= AUTO
GO           ?= go
JOBS         ?= $(shell nproc 2>/dev/null || echo 4)

# Never download a Go toolchain; go.mod asks for Go >= 1.24.
export GOTOOLCHAIN ?= local

.PHONY: all engine go test test-engine test-go e2e fmt vet clean

all: engine go

engine:
	cmake -S engine -B $(ENGINE_BUILD) -DCMAKE_BUILD_TYPE=Release -DFUSION_CUDA=$(CUDA)
	cmake --build $(ENGINE_BUILD) -j $(JOBS)

go:
	$(GO) build -o bin/ ./cmd/...

test: test-engine test-go

test-engine: engine
	$(ENGINE_BUILD)/fusion_tests
	$(ENGINE_BUILD)/rtier_node_tests

test-go: engine
	RTIER_ENGINE_BIN=$(abspath $(ENGINE_BUILD)) $(GO) test -race ./...

e2e: engine
	RTIER_ENGINE_BIN=$(abspath $(ENGINE_BUILD)) $(GO) test -race -count=1 -v ./test/e2e/

fmt:
	gofmt -w cmd internal test

vet:
	$(GO) vet ./...

clean:
	rm -rf bin $(ENGINE_BUILD)
