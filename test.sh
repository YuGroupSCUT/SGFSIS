# MoNuSAC-->CoNSeP
python 04_test.py \
--config "/home/data1/my/Project/SGFSL/SGFSL_ours/config/tsk_finetuning/ext_monusac_tsk_consep.yaml"

# MoNuSAC-->Lizard
python 04_test.py \
--config "/home/data1/my/Project/SGFSL/SGFSL_ours/config/tsk_finetuning/ext_monusac20x_tsk_lizard.yaml"

# CoNSeP-->MoNuSAC
python 04_test.py \
--config "/home/data1/my/Project/SGFSL/SGFSL_ours/config/tsk_finetuning/ext_consep_tsk_monusac.yaml"

# PanNuke-->MoNuSAC
python 04_test.py \
--config "/home/data1/my/Project/SGFSL/SGFSL_ours/config/tsk_finetuning/ext_pannuke_tsk_monusac.yaml"
