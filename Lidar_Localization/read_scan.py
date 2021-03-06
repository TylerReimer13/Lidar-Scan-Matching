import matplotlib.pyplot as plt
from scan_matching import ICP
from test_data import *
from math import acos


class ScanReader:
    def __init__(self, angle_min, angle_max, angle_inc, range_min, range_max):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_inc = angle_inc
        self.range_min = range_min
        self.range_max = range_max

    @staticmethod
    def rotation_matrix(ang, direction="cw"):
        d = 1
        if direction == "cw":
            d = -1
        elif direction == "ccw":
            d = 1
        else:
            print("Rotation direction not valid. Needs to be 'cw' or 'ccw'")

        r = np.array([[np.cos(ang), -d * np.sin(ang)],
                      [d * np.sin(ang), np.cos(ang)]])

        return r

    def convert_rng_to_xy(self, scan_ranges, rotation=0., loc=np.zeros(2)):
        curr_angle = self.angle_min
        xy_pts = []

        R = self.rotation_matrix(rotation)

        for rng in scan_ranges:
            if rng >= self.range_max:
                curr_angle += self.angle_inc
                continue

            x = rng * np.cos(curr_angle) + loc[0]
            y = rng * np.sin(curr_angle) + loc[1]

            xy_arr = np.array([x, y])
            xy_rot = R @ xy_arr

            xy_pts.append(xy_rot)

            curr_angle += self.angle_inc

        return np.array(xy_pts)

    def add_plot(self, points, marker='bX', markersize=1., label=None):
        plt.plot([pt[0] for pt in points], [pt[1] for pt in points], marker, markersize=markersize, label=label)

    def show_plot(self, xlims, ylims):
        plt.xlim(xlims[0], xlims[1])
        plt.ylim(ylims[0], ylims[1])
        plt.grid()
        plt.legend()
        plt.show()


class OccupancyGrid:
    def __init__(self, n_rows, n_cols, x_lims, y_lims):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.x_lims = x_lims
        self.y_lims = y_lims

        self.grid = np.zeros((self.n_rows, self.n_cols))

        self.x_incr = ((self.x_lims[1] - self.x_lims[0]) / self.n_cols) / 2
        self.y_incr = ((self.y_lims[1] - self.y_lims[0]) / self.n_rows) / 2

    def getGrid(self):
        return self.grid.copy()

    def get_cell_center(self, cell_coords):
        x = (self.x_lims[0] + self.x_incr) + (cell_coords[0] * 2 * self.x_incr)
        y = -((self.y_lims[0] + self.y_incr) + (cell_coords[1] * 2 * self.y_incr))
        return [x, y]

    def xy_to_cell(self, pt):
        for x in range(self.n_cols):
            for y in range(self.n_rows):
                center_x, center_y = self.get_cell_center([x, y])
                if (center_x - self.x_incr) < pt[0] <= (center_x + self.x_incr):
                    if (center_y - self.y_incr) < pt[1] <= (center_y + self.y_incr):
                        return x, y

    def fill_cell(self, xy_coords):
        for coord in xy_coords:
            x, y = self.xy_to_cell(coord)
            self.grid[y][x] = 255.


if __name__ == "__main__":
    reader = ScanReader(ang_min, ang_max, ang_inc, rng_min, rng_max)

    # Read in original point cloud scan, and leave it as is
    pos1 = np.array([0., 0.])
    pts1 = reader.convert_rng_to_xy(ranges3, 0., np.zeros(2))

    # Make a copy of first point cloud, and rotate and translate
    pos2 = np.array([0., 0.])
    pts2 = reader.convert_rng_to_xy(ranges4, 0., np.zeros(2))

    # Show Original Scans #
    reader.add_plot(pts1, 'bo', 1., label="Scan1")
    reader.add_plot(pts2, 'ro', 1., label="Scan2")
    reader.show_plot((-5., 5.), (-5., 5.))

    icp = ICP()

    T, transformed = icp(pts2, pts1, n_iter=20)

    print("Final Transformation Matrix")
    print(T)

    print("Translation: ", -T[:2, 2])
    print("Rotation: ", acos(T[0, 0]))

    plt.plot(pts1[:, 0], pts1[:, 1], 'bX', alpha=.35, label="First Scan")
    # plt.plot(pts2[:, 0], pts2[:, 1], 'gX', alpha=.25, label="Second Scan")
    plt.plot(transformed[:, 0], transformed[:, 1], 'rX', alpha=.5, label="Calculated")
    plt.grid()
    plt.xlim(-5., 5.)
    plt.ylim(-5., 5.)
    plt.legend()
    plt.show()

    # Make Occupancy Grid (Not Used Yet) #
    # occ_grid = OccupancyGrid(125, 125, [-10., 10.], [-10., 10.])
    # occ_grid.fill_cell(pts)
    # grid = occ_grid.getGrid().astype('uint8')


