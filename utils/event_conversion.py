import torch

def trivial_conversion(image, timesteps):
    events = []

    for _ in range(timesteps):
        events.append(image.clone())

    return events

def random_conversion(image, timesteps):
    events = []

    for _ in range(timesteps):
        temp = image.clone()
        temp[image > torch.rand_like(image)] = 0
        events.append(temp)

    return events