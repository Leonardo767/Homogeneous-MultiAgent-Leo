import numpy as np
import minecraft_H8 as minecraft
import time


def build_world0():
    world_shape = (10, 8, 10)
    world = np.zeros(world_shape)
    # place source blocks
    for pos in minecraft.SOURCES:
        world[pos[0], pos[1], pos[2]] = -2
        if pos[0]-1 >= 0:
            world[pos[0]-1, pos[1], pos[2]] = 0
        if pos[0]+1 < world_shape[0]:
            world[pos[0]+1, pos[1], pos[2]] = 0
        if pos[2]-1 >= 0:
            world[pos[0], pos[1], pos[2]-1] = 0
        if pos[2]+1 < world_shape[2]:
            world[pos[0], pos[1], pos[2]+1] = 0
    # place agents
    rx, ry, rz = np.random.randint(world_shape[0]), np.random.randint(
        2), np.random.randint(world_shape[2])
    while not (world[rx, ry, rz] == 0 and ((ry == 0) or (ry > 0 and world[rx, ry-1, rz] == -1))):
        rx, ry, rz = np.random.randint(world_shape[0]), np.random.randint(
            world_shape[1]), np.random.randint(world_shape[2])
    world[rx, ry, rz] = 1
    return world


num_workers = 1
FULL_HELP = True
MAP_ID = 9

gameEnv = minecraft.MinecraftEnv(num_workers, observation_range=-1, world0=build_world0(),
                                 observation_mode='default', FULL_HELP=FULL_HELP, MAP_ID=MAP_ID)
# gameEnv._render()
# time.sleep(1)
print(gameEnv.world.state)
# _ = input('Press key')
