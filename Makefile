run-training-dev:
	cargo run --bin train

run-api-dev:
	cargo run --bin api

run-request-health:
	curl http://localhost:8080/health
