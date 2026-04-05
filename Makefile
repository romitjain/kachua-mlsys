PYTHON ?= .venv/bin/python
MODAL ?= modal
NCU ?= ncu --set detailed --target-processes all --launch-skip 6 --launch-count 1 --kernel-name-base function --kernel-name "regex:^gdn_.*"
NCU_UI ?= ncu-ui
WORKSPACE ?= .
INPUT ?=
OUTPUT ?= profile_local
OUT_DIR ?= $(WORKSPACE)/profiles
REPORT := $(OUT_DIR)/$(OUTPUT).ncu-rep
WORKLOAD_IDX ?= 0
PROFILE_ARGS ?=
PROFILE_INPUT_ARG := $(if $(strip $(INPUT)),--input "$(INPUT)",)

.DEFAULT_GOAL := profile

.PHONY: profile profile-ui bench-local bench-modal timing-local timing-modal

profile:
	mkdir -p "$(OUT_DIR)"
	$(NCU) -o "$(OUT_DIR)/$(OUTPUT)" $(PYTHON) scripts/profile_local.py --workspace "$(WORKSPACE)" $(PROFILE_INPUT_ARG) $(PROFILE_ARGS)
	@echo 'Saved the profile: $(OUT_DIR)/$(OUTPUT)'

profile-ui:
	mkdir -p "$(OUT_DIR)"
	$(NCU) -o "$(OUT_DIR)/$(OUTPUT)" $(PYTHON) scripts/profile_local.py --workspace "$(WORKSPACE)" $(PROFILE_INPUT_ARG) $(PROFILE_ARGS)
	@echo 'Opening $(REPORT) in Nsight Compute UI'
	@nohup $(NCU_UI) "$(REPORT)" >/dev/null 2>&1 &

bench-local:
	$(PYTHON) scripts/run_local.py --workspace "$(WORKSPACE)"

bench-modal:
	$(MODAL) run scripts/run_modal.py --workspace "$(WORKSPACE)"

timing-local:
	$(PYTHON) scripts/bench_fi_timing.py --workspace "$(WORKSPACE)" --workload-idx $(WORKLOAD_IDX)

timing-modal:
	$(MODAL) run scripts/bench_fi_timing_modal.py --workspace "$(WORKSPACE)" --workload-idx $(WORKLOAD_IDX)
