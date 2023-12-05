all: 

check-env-var = $(if $(strip $($(1))),,$(error Environment variable $(1) is not defined))

deploy-grpc-yolo:
	$(call check-env-var,YOLO_SERVICE_PORT)
	python3 serving/yolo/yolo_service.py

deploy-http-router:
	$(call check-env-var,ROUTER_SERVICE_PORT)
	python3 serving/router/router.py

deploy-typefly:
	$(call check-env-var,TYPEFLY_SERVICE_IP)
	$(call check-env-var,YOLO_SERVICE_PORT)
	python3 serving/webui/typefly.py