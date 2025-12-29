.PHONY: serving_stop, serving_start, serving_remove, serving_open, serving_build

GPU_DEVICES=0
GPU_OPTIONS=$(shell if [ -f /proc/driver/nvidia/version ]; then echo "--gpus all -e CUDA_VISIBLE_DEVICES=$(GPU_DEVICES)"; else echo ""; fi)

CONTAINER_NAME=typefly-serving

serving_stop:
	@echo "=> Stopping $(CONTAINER_NAME)..."
	@-docker stop -t 0 $(CONTAINER_NAME) > /dev/null 2>&1
	@-docker rm -f $(CONTAINER_NAME) > /dev/null 2>&1

serving_start:
	@make serving_stop
	@echo "=> Starting $(CONTAINER_NAME)..."
	docker run -td --privileged --net=host $(GPU_OPTIONS) --ipc=host \
		--env-file ./docker/env.list \
    	--name="$(CONTAINER_NAME)" $(CONTAINER_NAME):0.1

serving_remove:
	@echo "=> Removing $(CONTAINER_NAME)..."
	@-docker image rm -f $(CONTAINER_NAME):0.1  > /dev/null 2>&1
	@-docker rm -f $(CONTAINER_NAME) > /dev/null 2>&1

serving_open:
	@echo "=> Opening bash in $(CONTAINER_NAME)..."
	@docker exec -it $(CONTAINER_NAME) bash

serving_build:
	@echo "=> Building $(CONTAINER_NAME)..."
	@make serving_stop
	@make serving_remove
	@echo -n "=>"
	docker build -t $(CONTAINER_NAME):0.1 -f ./docker/Dockerfile .
	@echo -n "=>"
	@make serving_start