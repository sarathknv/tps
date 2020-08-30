"""
Usage:
    python3 tps.py

    Use left mouse button click in the 2D subplot (bottom subplot)
    to put a control point there.

    Clicking on the same location clears the control point.

    Plots are regenerated if there's a change in control points.

    (Image that is being fitted is stored in 'img')
"""
import argparse
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.ion()


class TPS2d:
    def __init__(self):
        """
        Note: Variable notaton is the same as that in the Principal Warps
        paper and https://profs.etsmtl.ca/hlombaert/thinplates/
        """
        self.a = None  # variables of the linear part of the TPS equation
        self.W = None  # variables of the non-linear part of the TPS equation
        self.cps = None  # control point locations [(x1, y1), (x2, y2), ...]

    def _f(self, locs):
        linear_part = self.a[0] + np.dot(locs, self.a[1:])
        bending_part = np.dot(self._U(cdist(locs, self.cps)), self.W)
        return linear_part + bending_part

    def f(self, x, y):
        """Computes the height at the given set of positions x, y
        Usage:
            x, y = [0, 1, 2], [2, 3, 0]
            tps.f(x, y)

            # for single position
            x, y = [0,], [1,]
        """
        locs = np.vstack([x, y]).T
        return self._f(locs)

    def fit(self, v, cps):
        """Solves the TPS variables (W and a) for a given set of control points
        cps: control point locations of the form [(x1, y1), (x2, y2), ...]
        v: height at the control points [v1, v2, ...]

        Solves the equation and updates the self.a and self.W variables

        Code is borrowed from https://github.com/mdedonno1337/TPS/blob/master/TPS/TPSpy/__init__.py
        """
        self.cps = cps

        n = len(v)

        K = self._U(cdist(cps, cps, metric='euclidean'))
        P = np.hstack([np.ones((n, 1)), cps])

        L = np.vstack([np.hstack([K, P]), np.hstack([P.T, np.zeros((3, 3))])])

        Y = np.hstack([v, np.zeros((3,))])

        Wa = solve(L, Y)

        self.W = Wa[:-3].copy()
        self.a = Wa[-3:].copy()

    @staticmethod
    @np.vectorize
    def _U(r):
        if r == 0.0:
            return 0.0
        else:
            return (r**2) * np.log(r**2)


def onclick(event):
    """Updates the x, y variables with the selected location.
    Only clicks in the subplot2 i.e., ax2 are valid.
    """
    global x, y, clicked, ax2
    if event.inaxes == ax2:
        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        clicked = True

        # Clear current plots when clicked
        # Updation with new plots is done in the loop
        ax1.clear()
        ax2.clear()
        _subplots_label_stuff()


def plot_cps(v, cps):
    # If number of  initial controls points are changed, change 4 to the number
    # of initial controls points
    for i in range(4, len(v)):
        x, y = cps[i]
        ax1.plot((x, x), (y, y), (0, v[i]))
        ax2.plot(x, y, 'o')


def _subplots_label_stuff():
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid()
    ax2.set_xticks(np.arange(-0.5, img.shape[1], 1))
    ax2.set_yticks(np.arange(-0.5, img.shape[0], 1))
    ax2.set_xticklabels(np.arange(0, img.shape[1]+1, 1))
    ax2.set_yticklabels(np.arange(0, img.shape[0]+1, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', required=True, help='Path to image')
    parser.add_argument('--shape', required=False, default=(16, 16),
                        help='Resize the image')
    args = parser.parse_args()

    global x, y, clicked, ax1, ax2, img
    x, y = 0, 0
    clicked = False  # left mouse button click

    img = cv2.imread(args.path, cv2.IMREAD_GRAYSCALE)
    if args.shape:
        img = cv2.resize(img, dsize=args.shape, interpolation=cv2.INTER_CUBIC)

    # Initialize TPS
    # Surface approaches xy plane at infinity
    tps = TPS2d()
    v = [0, 0, 0, 0]
    cps = [(-1000, -1000), (1000, 1000), (-1000, 1000), (1000, -1000)]
    tps.fit(v, cps)

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         cps.append((i, j))
    #         v.append(img[i][j])
    # tps.fit(v, cps)

    # Plotting and mouse event stuff
    fig = plt.figure()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    ax1 = fig.add_subplot(2, 1, 1, projection='3d')  # TPS surface in 3D
    ax2 = fig.add_subplot(2, 1, 2)  # 2D image that's being fitted
    _subplots_label_stuff()

    X, Y = np.meshgrid(np.linspace(0, img.shape[1], img.shape[1]*2),
                       np.linspace(0, img.shape[0], img.shape[0]*2))
    _X = X.reshape(-1)
    _Y = Y.reshape(-1)
    _Z = tps.f(_X, _Y)
    Z = _Z.reshape(X.shape)

    # Initial surface plot, going to be the xy plane
    # ax1.contour3D(X, Y, Z, 50, cmap='binary')
    ax1.plot_wireframe(X, Y, Z, cmap='binary')
    ax2.imshow(img, cmap='gray', interpolation=None)

    plot_cps(v, cps)

    while True:
        if clicked:
            clicked = False
            if (x, y) not in cps:
                # If new point not in cps (control points), add it to the list
                v.append(img[y, x])
                cps.append((x, y))

            else:
                # If point in cps, remove it
                id = cps.index((x, y))
                v.pop(id)
                cps.pop(id)

            # Fit new TPS
            tps.fit(v, cps)

            # Plot the new surface. Old one is cleared in the callback function
            _Z = tps.f(_X, _Y)
            Z = _Z.reshape(X.shape)
            # ax1.contour3D(X, Y, Z, 50, cmap='binary')
            ax1.plot_wireframe(X, Y, Z, cmap='binary')
            ax2.imshow(img, cmap='gray', interpolation=None)

            # Also plot the control points in both the subplots
            plot_cps(v, cps)

        fig.canvas.draw()
        plt.pause(0.5)
