
projectName = jarvis-v2

snapshotVersion = 0.0.1
lastReleaseVersion = 0.0.0

dockerRepository = 304848207041.dkr.ecr.eu-central-1.amazonaws.com
dockerImage = sumgpt/latest

# Targets

.PHONY: setup tensorboard update compile test run release deploy

setup:
	$(call output, "SETUP $(projectName)")
	@python --version
	@pip --version
	@pyenv --version
	@pytest --version

	@if test -f ".python-version"; then \
		echo ".python-version exists."; \
	else \
		echo "pyenv not set for local directory - set using 'pyenv local'"; \
		exit 1; \
	fi

tensorboard:
	echo http://localhost:6006/?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_all_f1%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_all_FPR%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_all_FNR%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_loss%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22train_loss%22%7D%5D&darkMode=true#timeseries
	tensorboard --logdir ./output

update:
	$(call output, "UPDATE $(projectName)")
	@pip install -r requirements.txt

test: update
	$(call output, "TESTS $(projectName) v$(snapshotVersion)")
	@pip install pytest
	@python -m pytest

run: test
	$(call output, "RUN $(projectName) v$(snapshotVersion)")
	@echo Hello world!

build-docker-local:
	$(call output, "BUILD DOCKER LOCAL")
	@docker build -t $(projectName) .

run-docker-local: build-docker-local
	$(call output, "RUN DOCKER LOCAL")
	@docker run --rm -p 5000:80 $(projectName) .

build-docker:
	$(call output, "BUILD DOCKER $(projectName) v$(snapshotVersion)")
	@docker build -t $(dockerRepository)/$(dockerImage):$(snapshotVersion) .

run-docker: build-docker
	$(call output, "RUN DOCKER $(projectName) v$(snapshotVersion)")
	@docker run --rm -p 5000:80 $(dockerRepository)/$(dockerImage):$(snapshotVersion)

release: build-docker test
	$(call output, "RELEASE $(projectName) v$(snapshotVersion)")

	$(call validateStartedFromBranch)
	$(call configureGitUser)

	@eval $(shell aws ecr get-login --no-include-email)
	@docker push $(dockerRepository)/$(dockerImage):$(snapshotVersion)

	@./scripts/bumpVersion.sh $(snapshotVersion) $(lastReleaseVersion)

	@git add Makefile
	@git commit -m "- Release v$(snapshotVersion) [skip ci]"
	@git push

	@git tag v$(snapshotVersion)
	@git push origin v$(snapshotVersion)

deploy:
	$(call output, "DEPLOY $(projectName) v$(lastReleaseVersion) to $(env) environment")
	$(call validateStartedFromBranch)
	@./scripts/deploy.sh $(env) $(dockerRepository)/$(dockerImage) $(lastReleaseVersion)
	$(call configureGitUser)

	@git tag $(deployTag)
	@git push origin $(deployTag)

define output
	@echo "---------------------------------------------------------------\n$(1) \n---------------------------------------------------------------"
endef

define configureGitUser
	@if [ "$(CI)" = true ]; then \
		git config user.email "dev@talentflyxpert.com"; \
		git config user.name "Pipeline User"; \
	fi
endef

define validateStartedFromBranch
	@if [ "$(CI)" = true ] && [ -z "$(BITBUCKET_BRANCH)" ]; then \
      echo "ERROR: HEAD is detached from origin, did you start the pipeline from a commit?"; \
      exit 1; \
    fi
endef