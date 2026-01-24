# python src/routing_runner.py \
#   --problem tsp --mode rlho --nodes 100 --policy outputs/tsp100.pt \
#   --rl-steps 128 --sa-t0 5 \
#   --sa-converge --sa-steps 50000 --sa-stall-steps 2000 --sa-alpha 0.995 \
#   --episodes 1 --num-envs 1 --sa-log-interval 0 --log-interval 1 > test_tsp100.log

export CUDA_VISIBLE_DEVICES=5

# python src/routing_runner.py \
#   --problem tsp --mode rlho --nodes 50 --policy outputs/tsp50.pt \
#   --rl-steps 128 --sa-t0 5 \
#   --sa-converge --sa-steps 50000000 --sa-alpha 0.9999995993977071 \
#   --episodes 1000 --num-envs 256 --sa-log-interval 100000 --sa-log-env 0

python src/routing_runner.py \
  --problem tsp --mode rlho --nodes 100 --policy outputs/tsp100.pt \
  --rl-steps 128 --sa-t0 5 \
  --sa-converge --sa-steps 50000000 --sa-alpha 0.9999995993977071 \
  --episodes 1000 --num-envs 256 --sa-log-interval 100000 --sa-log-env 0

python src/routing_runner.py \
  --problem cvrp --mode rlho --customers 100 --capacity 50 --policy outputs/cvrp100.pt \
  --rl-steps 128 --sa-t0 5 \
  --sa-converge --sa-steps 50000000 --sa-alpha 0.9999995993977071 \
  --episodes 1000 --num-envs 256 --sa-log-interval 100000 --sa-log-env 0

python src/routing_runner.py \
  --problem cvrp --mode rlho --customers 50 --capacity 40 --policy outputs/cvrp50.pt \
  --rl-steps 128 --sa-t0 5 \
  --sa-converge --sa-steps 50000000 --sa-alpha 0.9999995993977071 \
  --episodes 1000 --num-envs 256 --sa-log-interval 100000 --sa-log-env 0