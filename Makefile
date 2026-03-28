NCU ?= ncu --set detailed --target-processes all --launch-skip 6 --launch-count 1 --kernel-name-base function --kernel-name "regex:^gdn_.*"
NCU_UI ?= ncu-ui
INPUT ?= solution/cuda/kernel.cu
OUTPUT ?= profile_local
OUT_DIR ?= profiles
REPORT := $(OUT_DIR)/$(OUTPUT).ncu-rep

.DEFAULT_GOAL := profile

.PHONY: profile
profile:
	mkdir -p "$(OUT_DIR)"
	$(NCU) -o "$(OUT_DIR)/$(OUTPUT)" python scripts/profile_local.py --input "$(INPUT)"
	@echo 'Opening $(REPORT) in Nsight Compute UI'
	@nohup $(NCU_UI) "$(REPORT)" >/dev/null 2>&1 &
