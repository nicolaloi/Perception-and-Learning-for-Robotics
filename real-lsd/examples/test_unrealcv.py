import json
from unrealcv import client
from unrealcv.util import read_npy, read_png
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import glog as log

help_message = '''
A demo showing how to control a game using python
a, d: rotate camera to left and right.
q, e: move camera up and down.
left, right, up, down: move around
'''
plt.rcParams['keymap.save'] = ''


def main():
    loc = None
    rot = None

    fig, ax = plt.subplots()
    img = np.zeros((480, 640, 4))
    ax.imshow(img)

    def onpress(event):
        rot_offset = 10 # Rotate 5 degree for each key press
        loc_offset = 10 # Move 5.0 when press a key

        # Up and Down in cam-plane
        if event.key == 'w': loc[0] += loc_offset
        if event.key == 's': loc[0] -= loc_offset
        # Left and Right movement in cam-plane
        if event.key == 'a': loc[1] -= loc_offset
        if event.key == 'd': loc[1] += loc_offset
        # In and Out movement into cam-plane
        if event.key == 'q': loc[2] += loc_offset
        if event.key == 'e': loc[2] -= loc_offset

        # cmd = 'vset /camera/0/rotation %s' % ' '.join([str(v) for v in rot])
        # client.request(cmd)
        cmd = 'vset /camera/0/moveto %s' % ' '.join([str(v) for v in loc])
        client.request(cmd)

        print(client.request('vget /camera/0/location'))
        print(client.request('vget /camera/0/rotation'))
        res = client.request('vget /camera/2/lit png')
        img = read_png(res)

        ax.imshow(img)
        fig.canvas.draw()

    client.connect()
    if not client.isconnected():
        print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
        return
    else:
        print(help_message)

    print("-------------------------------------------------------------------")
    print(client.request('vget /objects'))
    print("-------------------------------------------------------------------")
    print(client.request('vget /cameras'))
    print("-------------------------------------------------------------------")
    # print(client.request('vget /cameras'))
    # print("-------------------------------------------------------------------")

    init_loc = [float(v) for v in client.request('vget /camera/0/location').split(' ')]
    init_rot = [float(v) for v in client.request('vget /camera/0/rotation').split(' ')]

    loc = init_loc; rot = init_rot

    fig.canvas.mpl_connect('key_press_event', onpress)
    plt.title('Keep this window in focus, it will be used to receive key press event')
    plt.axis('off')
    plt.show() # Add event handler

if __name__ == '__main__':
    main()
