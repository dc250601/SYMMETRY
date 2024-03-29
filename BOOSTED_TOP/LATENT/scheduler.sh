#!/bin/bash
#SBATCH -A m4392
#SBATCH --constraint=gpu
#SBATCH -c 128
#SBATCH --mem=0
#SBATCH --qos=preempt
#SBATCH --time=24:00:00
#SBATCH -n 1
#SBATCH --job-name=Boosted-Top-channel"-$1"
#SBATCH --requeue

unzip -q /pscratch/sd/d/diptarko/Top_full.zip -d /dev/shm/

start=$((10#$(date +%H)+$((10#$(date +%j)*24))))

timeout 23h python3 trainer.py 0 0 200 $1 &
timeout 23h python3 trainer.py 1 0 200 $1 &
timeout 23h python3 trainer.py 2 1 200 $1 &
timeout 23h python3 trainer.py 3 1 200 $1 &
timeout 23h python3 trainer.py 4 2 200 $1 &
timeout 23h python3 trainer.py 5 2 200 $1 &
timeout 23h python3 trainer.py 6 3 200 $1 &
timeout 23h python3 trainer.py 7 3 200 $1 &

wait

end=$((10#$(date +%H)+$((10#$(date +%j)*24))))

tot=$((end-start))

if [ $tot -gt 2 ]; then
	sbatch pscratch/sd/d/diptarko/SYMMETRY/BOOSTED_TOP/LATENT/scheduler.sh $1
fi