# Expects repo venv at ../.venv; override PYTHON= if different.
PYTHON ?= ../.venv/bin/python
MODAL ?= modal
NCU ?= ncu --set detailed --target-processes all --launch-skip 6 --launch-count 1 --kernel-name-base function --kernel-name "regex:^gdn_.*"
NCU_UI ?= ncu-ui
INPUT ?= solution/cuda/kernel.cu
OUTPUT ?= profile_local
OUT_DIR ?= profiles
REPORT := $(OUT_DIR)/$(OUTPUT).ncu-rep
WORKLOAD_IDX ?= 0

.DEFAULT_GOAL := profile

.PHONY: profile profile-ui bench-local bench-modal timing-local timing-modal

profile:
	mkdir -p "$(OUT_DIR)"
	$(NCU) -o "$(OUT_DIR)/$(OUTPUT)" $(PYTHON) scripts/profile_local.py --input "$(INPUT)"
	@echo 'Saved the profile: $(OUT_DIR)/$(OUTPUT)'

profile-ui:
	mkdir -p "$(OUT_DIR)"
	$(NCU) -o "$(OUT_DIR)/$(OUTPUT)" $(PYTHON) scripts/profile_local.py --input "$(INPUT)"
	@echo 'Opening $(REPORT) in Nsight Compute UI'
	@nohup $(NCU_UI) "$(REPORT)" >/dev/null 2>&1 &

bench-local:
	$(PYTHON) scripts/run_local.py

bench-modal:
	$(MODAL) run scripts/run_modal.py

timing-local:
	$(PYTHON) scripts/bench_fi_timing.py --workload-idx $(WORKLOAD_IDX)

timing-modal:
	$(MODAL) run scripts/bench_fi_timing_modal.py --workload-idx $(WORKLOAD_IDX)
