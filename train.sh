export CUDA_VISIBLE_DEVICES=5

# nohup python ./src/routing_trainer.py outputs/tsp50.pt \
#     --problem tsp \
#     --nodes 50 \
#     --rl-steps 50 \
#     --sa-steps 200 \
#     --epochs 50 > tsp50.log 2>&1 &

python ./src/routing_trainer.py outputs/tsp50.pt \
    --problem tsp \
    --nodes 50 \
    --rl-steps 50 \
    --sa-steps 200 \
    --epochs 50

python ./src/routing_trainer.py outputs/tsp100.pt \
    --problem tsp \
    --nodes 100 \
    --rl-steps 50 \
    --sa-steps 200 \
    --epochs 50

python ./src/routing_trainer.py outputs/cvrp50.pt \
    --problem cvrp \
    --nodes 50 \
    --rl-steps 50 \
    --sa-steps 200 \
    --epochs 50

python ./src/routing_trainer.py outputs/cvrp100.pt \
    --problem cvrp \
    --nodes 100 \
    --rl-steps 50 \
    --sa-steps 200 \
    --epochs 50