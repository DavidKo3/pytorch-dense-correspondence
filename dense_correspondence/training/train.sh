#!/bin/bash

python train_simulated.py --name=rope_523_task_loops_halfsize_3 --dataset=rope_523_task_loops_halfsize_only.yaml
python train_simulated.py --name=rope_523_task_loops_quartersize_3 --dataset=rope_523_task_loops_quartersize_only.yaml

