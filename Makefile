.PHONY: edge_stop, edge_start, edge_remove, edge_open, edge_build

GPU_OPTIONS=--gpus all

ifneq ($(origin OPENAI_API_KEY), environment)
  $(error Environment variable OPENAI_API_KE is not defined)
endif

edge_stop:
	@echo "=> Stopping typefly-edge..."
	@-docker stop -t 0 typefly-edge > /dev/null 2>&1
	@-docker rm -f typefly-edge > /dev/null 2>&1

edge_start:
	@make edge_stop
	@echo "=> Starting typefly-edge..."
	docker run -td --privileged --net=host $(GPU_OPTIONS) --ipc=host \
		--env-file ./docker/env.list \
    	--name="typefly-edge" typefly-edge:0.1

edge_remove:
	@echo "=> Removing typefly-edge..."
	@-docker image rm -f typefly-edge:0.1  > /dev/null 2>&1
	@-docker rm -f typefly-edge > /dev/null 2>&1

edge_open:
	@echo "=> Opening bash in typefly-edge..."
	@docker exec -it typefly-edge bash

edge_build:
	@echo "=> Building typefly-edge..."
	@make edge_stop
	@make edge_remove
	@echo -n "=>"
	docker build -t typefly-edge:0.1 -f ./docker/edge/Dockerfile .
	@echo -n "=>"
	@make edge_start