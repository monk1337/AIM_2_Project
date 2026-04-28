# Reproducibility targets for AIM_2_Project.
#
# All paths assume the layout described in README.md and SETUP.md:
#   data/pov_surgery/, data/aria_val/, data/phase0_sidecars/, data/mano/
#   checkpoints/{wilor,hamer,handoccnet,meshgraphormer}/
#   results/
#
# Override paths by passing them on the command line:
#   make eval-wilor-pov DATA=/mnt/datasets RESULTS=/mnt/results

DATA       ?= data
CKPT       ?= checkpoints
RESULTS    ?= results
PYTHON     ?= python

.PHONY: help eval-wilor-pov eval-hamer-pov eval-handoccnet-pov eval-mgfm-pov \
        eval-wilor-aria eval-hamer-aria eval-handoccnet-aria eval-mgfm-aria \
        eval-ensemble eval-per-seq eval-per-finger \
        ft-wilor-pov ft-wilor-mixed ft-wilor-anchored ft-wilor-distill ft-honet \
        precompute-teacher viz-mesh viz-kpts viz-export

help:
	@echo "Targets (run from repo root):"
	@echo "  eval-{wilor,hamer,handoccnet,mgfm}-{pov,aria}   off-shelf evaluation"
	@echo "  eval-ensemble                                   weighted MGFM+HONet"
	@echo "  eval-per-seq                                    per-sequence breakdown"
	@echo "  eval-per-finger                                 per-finger PA-MPJPE"
	@echo "  ft-wilor-pov                                    Exp A (POV-only)"
	@echo "  ft-wilor-mixed                                  Exp B (POV + Aria HSAM)"
	@echo "  ft-wilor-anchored                               Exp B+ (mesh anchor)"
	@echo "  ft-wilor-distill                                Exp I  (MGFM+HONet -> WiLoR)"
	@echo "  ft-honet                                        HandOccNet FT (Exp E/F)"
	@echo "  precompute-teacher                              cache MGFM+HONet on Aria train"
	@echo "  viz-{mesh,kpts,export}                          rendering helpers"

# --- evaluation -------------------------------------------------------------

eval-wilor-pov:
	$(PYTHON) -m src.eval.eval_wilor_runner --dataset pov --data $(DATA) \
	    --ckpt $(CKPT)/wilor --out $(RESULTS)/wilor_pov.json

eval-wilor-aria:
	$(PYTHON) -m src.eval.eval_wilor_runner --dataset aria --data $(DATA) \
	    --ckpt $(CKPT)/wilor --out $(RESULTS)/wilor_aria.json

eval-hamer-pov:
	$(PYTHON) -m src.eval.eval_hamer --dataset pov --data $(DATA) \
	    --ckpt $(CKPT)/hamer --out $(RESULTS)/hamer_pov.json

eval-hamer-aria:
	$(PYTHON) -m src.eval.eval_hamer --dataset aria --data $(DATA) \
	    --ckpt $(CKPT)/hamer --out $(RESULTS)/hamer_aria.json

eval-handoccnet-pov:
	$(PYTHON) -m src.eval.eval_handoccnet --dataset pov --data $(DATA) \
	    --ckpt $(CKPT)/handoccnet --out $(RESULTS)/handoccnet_pov.json

eval-handoccnet-aria:
	$(PYTHON) -m src.eval.eval_handoccnet --dataset aria --data $(DATA) \
	    --ckpt $(CKPT)/handoccnet --out $(RESULTS)/handoccnet_aria.json

eval-mgfm-pov:
	$(PYTHON) -m src.eval.eval_meshgraphormer --dataset pov --data $(DATA) \
	    --ckpt $(CKPT)/meshgraphormer --out $(RESULTS)/mgfm_pov.json

eval-mgfm-aria:
	$(PYTHON) -m src.eval.eval_meshgraphormer --dataset aria --data $(DATA) \
	    --ckpt $(CKPT)/meshgraphormer --out $(RESULTS)/mgfm_aria.json

eval-ensemble:
	$(PYTHON) -m src.eval.eval_ensemble --w_mgfm 0.75 --w_honet 0.25 \
	    --data $(DATA) --ckpt $(CKPT) --out $(RESULTS)/ensemble.json

eval-per-seq:
	$(PYTHON) -m src.eval.eval_per_seq --data $(DATA) --ckpt $(CKPT) \
	    --out $(RESULTS)/per_seq_expB.json

eval-per-finger:
	$(PYTHON) -m src.eval.per_finger_eval --data $(DATA) --ckpt $(CKPT) \
	    --out $(RESULTS)/per_finger.json

# --- training ---------------------------------------------------------------

ft-wilor-pov:
	$(PYTHON) -m src.train.ft_wilor --data $(DATA) --ckpt $(CKPT)/wilor \
	    --out $(CKPT)/wilor_ft_pov

ft-wilor-mixed:
	$(PYTHON) -m src.train.ft_wilor_mixed --data $(DATA) --ckpt $(CKPT)/wilor \
	    --out $(CKPT)/wilor_ft_mixed

ft-wilor-anchored:
	$(PYTHON) -m src.train.ft_wilor_anchored --data $(DATA) --ckpt $(CKPT)/wilor \
	    --lambda_mesh 0.5 --out $(CKPT)/wilor_ft_anchored

ft-wilor-distill: precompute-teacher
	$(PYTHON) -m src.train.ft_wilor_distill --data $(DATA) --ckpt $(CKPT)/wilor \
	    --teacher_cache $(RESULTS)/teacher_cache.npz \
	    --out $(CKPT)/wilor_ft_distill

ft-honet:
	$(PYTHON) -m src.train.ft_honet --data $(DATA) --ckpt $(CKPT)/handoccnet \
	    --out $(CKPT)/honet_ft_mixed

precompute-teacher:
	$(PYTHON) -m src.train.precompute_teacher --data $(DATA) --ckpt $(CKPT) \
	    --out $(RESULTS)/teacher_cache.npz

# --- visualization ----------------------------------------------------------

viz-mesh:
	$(PYTHON) -m src.viz.viz_seq_mesh --data $(DATA) --ckpt $(CKPT) \
	    --out viz_seq_mesh/

viz-kpts:
	$(PYTHON) -m src.viz.viz_seq_kpts --data $(DATA) --ckpt $(CKPT) \
	    --out viz_seq_kpts/

viz-export:
	$(PYTHON) -m src.viz.export_mesh_data_v2 --data $(DATA) --ckpt $(CKPT) \
	    --out $(RESULTS)/mesh_export.json
