IMAGE_NAME = ml-docker-app
CONTAINER_NAME = ml-app
build:
	docker build -t $(IMAGE_NAME) .
run:
	docker run -d -p 5000:5000 --name $(CONTAINER_NAME) $(IMAGE_NAME)
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true
