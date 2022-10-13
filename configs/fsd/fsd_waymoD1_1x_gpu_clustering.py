_base_ = [
    './fsd_waymoD1_1x.py'
]


# Note that the output of current version of gpu_clustering
# is a little different from the CPU version, leading to performance
# drop in vehicle and cyclist around ~0.3 AP and 0.2 AP improve in pedestrian.
# We will fix it later.
model=dict(
    cluster_assigner=dict(
        gpu_clustering=(False, True), # only use gpu version during inference
    ),
)