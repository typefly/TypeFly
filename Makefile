.PHONY: stop, start, remove, open, build

GPU_OPTIONS=--gpus all

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

typefly:
	bash ./serving/webui/install_requirements.sh
	cd ./proto && bash generate.sh
	python3 ./serving/webui/typefly.py --use_virtual_robot