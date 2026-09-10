IMAGE ?= ghcr.io/wojciechdudek/faceshim
TAG   ?= 0.1.3
PLATFORM ?= linux/amd64

.PHONY: build build-local push run selfcheck logs

## Buduje obraz produkcyjny dla hosta (domyslnie linux/amd64) i wypycha do rejestru.
build:
	docker buildx build --platform $(PLATFORM) -t $(IMAGE):$(TAG) -t $(IMAGE):latest --push .

## Buduje obraz na lokalna architekture, bez wypychania (do testow).
build-local:
	docker build -t $(IMAGE):dev .

push:
	docker push $(IMAGE):$(TAG) && docker push $(IMAGE):latest

run:
	docker run --rm --name faceshim -p 5002:5000 -v faceshim-data:/data $(IMAGE):dev

## Pelny test end-to-end na dzialajacym kontenerze, na zdjeciu testowym z insightface.
selfcheck:
	docker exec faceshim python selfcheck.py --image auto

logs:
	docker logs -f faceshim
