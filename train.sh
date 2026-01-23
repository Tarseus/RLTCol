export CUDA_VISIBLE_DEVICES=5

# nohup python ./src/routing_trainer.py outputs/tsp50.pt \
#     --problem tsp \
#     --nodes 50 \
#     --rl-steps 50 \
#     --sa-steps 1000 \
#     --epochs 50 > tsp50.log 2>&1 &

python ./src/routing_trainer.py outputs/tsp50.pt \
    --problem tsp \
    --nodes 50 \
    --rl-steps 128 \
    --sa-steps 1000 \
    --sa-t0 5.0 \
    --batch-size 128 \
    --pair-chunk-size 512 \
    --train-envs 4 \
    --step-per-collect 2000 \
    --epochs 50

# python ./src/routing_trainer.py outputs/tsp50.pt \
#     --problem tsp \
#     --nodes 100 \
#     --rl-steps 128 \
#     --sa-steps 1000 \
#     --sa-t0 5.0 \
#     --batch-size 128 \
#     --pair-chunk-size 512 \
#     --train-envs 4 \
#     --step-per-collect 2000 \
#     --epochs 50

# python ./src/routing_trainer.py outputs/cvrp50.pt \
#     --problem cvrp \
#     --customers 50 \
#     --rl-steps 128 \
#     --sa-steps 1000 \
#     --sa-t0 5.0 \
#     --batch-size 128 \
#     --pair-chunk-size 512 \
#     --train-envs 4 \
#     --step-per-collect 2000 \
#     --epochs 50

# python ./src/routing_trainer.py outputs/cvrp100.pt \
#     --problem cvrp \
#     --customers 100 \
#     --rl-steps 128 \
#     --sa-steps 1000 \
#     --sa-t0 5.0 \
#     --batch-size 128 \
#     --pair-chunk-size 512 \
#     --train-envs 4 \
#     --step-per-collect 2000 \
#     --epochs 50