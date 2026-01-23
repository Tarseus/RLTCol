# python src/routing_runner.py \
#   --problem tsp --mode rlho --nodes 100 --policy outputs/tsp100.pt \
#   --rl-steps 128 --sa-t0 5 \
#   --sa-converge --sa-steps 50000 --sa-stall-steps 2000 --sa-alpha 0.995 \
#   --episodes 1 --num-envs 1 --sa-log-interval 0 --log-interval 1 > test_tsp100.log

python src/routing_runner.py \
  --problem tsp --mode rlho --nodes 100 --policy outputs/tsp100.pt \
  --rl-steps 128 --sa-t0 5 \
  --sa-converge --sa-steps 50000 --sa-stall-steps 2000 \
  --episodes 1 --num-envs 1 --sa-log-interval 500 --sa-log-env 0

# python src/routing_runner.py \
#   --problem tsp --mode rlho --nodes 100 --policy outputs/tsp100.pt \
#   --rl-steps 128 --sa-t0 5 \
#   --sa-converge --sa-steps 50000 --sa-stall-steps 2000 \
#   --episodes 1000 --num-envs 16 --log-interval 10 \
#   --sa-log-interval 500

# python src/routing_runner.py \ 0.9999938
#   --problem cvrp --mode rlho --nodes 100 --policy outputs/cvrp100.pt \
#   --rl-steps 128 --sa-t0 5 \
#   --sa-converge --sa-steps 50000 --sa-stall-steps 2000

# python src/routing_runner.py \
#   --problem cvrp --mode rlho --nodes 50 --policy outputs/cvrp50.pt \
#   --rl-steps 128 --sa-t0 5 \
#   --sa-converge --sa-steps 50000 --sa-stall-steps 2000