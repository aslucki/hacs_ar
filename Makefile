IMAGE_NAME=hacs_ar
PORT ?= 8888

build:
	docker build -t $(IMAGE_NAME) .

dev:
	docker run --rm -ti  \
		--runtime=nvidia \
		-v $(PWD)/:/project \
		-v /media/storage/HACS-dataset/hacs_videos/:/project/videos \
		-v /media/storage/HACS-dataset/processed/:/project/videos_processed \
		-w '/project' \
		$(IMAGE_NAME)

lab:
	docker run --rm -ti  \
		-p $(PORT):$(PORT) \
		-v $(PWD)/:/project \
		-v /media/storage/HACS-dataset/processed/:/project/videos_processed \
		-w '/project' \
		$(IMAGE_NAME) \
		jupyter lab --ip=0.0.0.0 --port=$(PORT) --allow-root --no-browser

test:
	docker run --rm -ti  \
		-v $(PWD)/:/project \
		-w '/project' \
		$(IMAGE_NAME) \
		python3 -m pytest tests/ -s