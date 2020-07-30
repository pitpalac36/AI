import math


def huber_loss(realOutputs, computedOutputs, delta):
    absolute_diff = 0
    for i in range(len(realOutputs)):
        absolute_diff += abs(realOutputs[i] - computedOutputs[i])
    rmse = 0.5 * absolute_diff ** 2
    mae = delta * (absolute_diff - 0.5 * delta)
    if absolute_diff <= delta:
        return rmse
    return mae
