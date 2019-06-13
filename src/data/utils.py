import numpy as np


def depth_conversion(point_depth, w, h, f):
    """
    Converts from depth perspective to depth planner
    :param point_depth: depth perspective image
    :param w: image width
    :param h: image height
    :param f: camera focal value
    :return:
    """
    i_c = np.float(h) / 2 - 1
    j_c = np.float(w) / 2 - 1
    columns, rows = np.meshgrid(np.linspace(0, w - 1, num=w), np.linspace(0, h - 1, num=h))
    distance_from_center = ((rows - i_c) ** 2 + (columns - j_c) ** 2) ** 0.5
    return point_depth / (1 + (distance_from_center / f) ** 2) ** 0.5


def calc_focal_values(w, h, fov):
    """
    Calculates a cameras focal values
    :param w: image width
    :param h: image height
    :param fov: camera field of view (in degrees)
    :return:
    """
    cx = w / 2
    cy = h / 2
    f = w / (2 * (np.tan(fov * np.pi / 360)))
    return (cx, cy), f


def intrinsic_matrix_from_camera(w, h, fov):
    """
    Calculates the camera intrinsic matrix based on camera values
    :param w: image width
    :param h: image height
    :param fov: camera field of view (in degrees)
    :return:
    """
    (cx, cy), f = calc_focal_values(w, h, fov)
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def intrinsic_matrix_from_focal(cx, cy, f):
    """
    Calculates the camera intrinsic matrix based on a cameras focal values
    :param cx: x midpoint
    :param cy: y midpoint
    :param f: focal value
    :return:
    """
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def project_coordinate(depth_image, u, v, cx, cy, f):
    """

    :param depth_image: numpy depth planner image (in meters)
    :param u: image u pixel
    :param v: image v pixel
    :param cx: image x midpoint
    :param cy: image y midpoint
    :param f: camera focal value
    :return:
    """
    d = depth_image[v][u]
    x = (u - cx) * d / f
    y = (v - cy) * d / f
    z = d
    return x, y, z
