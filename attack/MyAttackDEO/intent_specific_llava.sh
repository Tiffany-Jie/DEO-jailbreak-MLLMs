# #!/bin/bash

# certain hyper-parameters can be modified based on user's preference

python intent_specific_llava.py \
    --output temp \
    --batch_size 1 \
    --num_samples 1000 \
    --steps 8 \
    --num_query 4\
    --num_sub_query 4 \
    --wandb \
    --wandb_project_name llava \
    --wandb_run_name temp