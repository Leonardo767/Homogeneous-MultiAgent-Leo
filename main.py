import numpy as np
import minecraft_H8 as minecraft
import time
from main_Vision import Window, convertEBtoSB


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
    # place agents randomly
    rx, ry, rz = np.random.randint(world_shape[0]), np.random.randint(
        2), np.random.randint(world_shape[2])
    while not (world[rx, ry, rz] == 0 and ((ry == 0) or (ry > 0 and world[rx, ry-1, rz] == -1))):
        rx, ry, rz = np.random.randint(world_shape[0]), np.random.randint(
            world_shape[1]), np.random.randint(world_shape[2])
    world[rx, ry, rz] = 1
    return world


def create_state_buffer(world):
    state = world.state.copy()
    for pos in minecraft.PLAN_MAPS[MAP_ID - 1]:
        if state[pos[0], pos[1], pos[2]] != -3:
            state[pos[0], pos[1], pos[2]] = -4
    state_buffer = convertEBtoSB(state)
    state_buffer = np.vstack((state_buffer, state_buffer))
    return state_buffer


num_workers = 1
FULL_HELP = True
MAP_ID = 9

gameEnv = minecraft.MinecraftEnv(num_workers, observation_range=-1, world0=build_world0(),
                                 observation_mode='default', FULL_HELP=FULL_HELP, MAP_ID=MAP_ID)
# gameEnv._render()
# print(gameEnv.world.state)

# render 3d
window = Window(width=640 + 160, height=480 + 120, caption='Pyglet',
                resizable=True, visible=False)
state_buffer = create_state_buffer(gameEnv.world)
window.set_State(state_buffer)
window.on_key_press(112, 0)
pos = (6, 4, 6)
rot = (-45, -45)
data = window.take_pics_3rd_person(pos, rot)
window.save_pics(data)
