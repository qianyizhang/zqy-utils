
import numbers
from scipy.spatial.transform import Rotation as R
import numpy as np


def norm(vec):
    """
    normalize the last dimension of a vector to unit 1.
    """
    v = np.linalg.norm(vec, axis=-1, keepdims=True)
    v[v < 1e-9] = 1.0   # guard against 0 division
    return vec / v


def get_rotation_matrix(rot_axis, angle):
    """
    get the rotation matrix for given axis + angle
    """
    rot_axis = norm(rot_axis)
    rad = np.deg2rad(angle)
    r = R.from_rotvec(rad * rot_axis)
    return r.as_dcm()


def get_range(pts, vec, size):
    """
    given a series of points and a vector, finds its range such
        all valid info are covered by pts + range* vec
    """
    vec[abs(vec) < 1e-9] = 1e-9
    pts = np.array(pts)
    intercepts = np.hstack([-pts / vec, (size - pts) / vec])
    intercepts.sort(-1)
    return min(intercepts[:, 2]), max(intercepts[:, 3])


def project_to_plane(pts, plane_norm):
    """
    given a plane_norm, return porjected points to that plane
    Args:
        pts (np.ndarray): Nx3 points
        plane_norm (3d vector): the normal vector
    Return:
        projected_pts (np.ndarray): Nx3 projected points
        distance (np.ndarray): N distance

    """
    plane_norm = norm(plane_norm)
    distance = pts.dot(plane_norm)
    projected_pts = pts - distance[:, None]*plane_norm
    return projected_pts, distance


def get_rotated_vec(rot_axis, angle, pivot_axis=(1, 0, 0)):
    """
    given a rotation axis, angle and pivot_axis, return the rotated vector
    """
    rot_axis = norm(rot_axis)
    rot_mat = get_rotation_matrix(rot_axis, angle)

    pivot_axis = norm(pivot_axis)
    vec = norm(np.cross(pivot_axis, rot_axis))
    rotated_vec = vec @ rot_mat
    return rotated_vec


def get_consistent_normals(pts, angle=0.0, pivot_axis=(1, 0, 0),
                           return_binormals=False, repeat_last=True):
    """
    get a series of normals (and binormals) from a series of points
    Args:
        pts (np.ndarray): Nx3 points
        angle (float): Default = 0.0. rotated angle around anchor vector,
            which is normal to rotation axis and pivot axis
        pivot_axis (3d vector): Default = (1, 0, 0)
        return_binormals (bool): Default = False
        repeat_last (bool): Default = True. Repeat last vector so normals
            (and binormals) have the same length as input points
    """
    tangents = norm(pts[1:] - pts[:-1])
    rot_axis = tangents[0]
    n0 = get_rotated_vec(rot_axis, angle, norm(pivot_axis))
    norm_list = [n0]
    """
    for t in tangents[1:]:
        tmp = np.cross(n0, t)
        n0 = np.cross(t, tmp)
        norm_list.append(n0)
    """

    def calc_norm(t):
        n0 = norm_list[-1]
        tmp = np.cross(n0, t)
        n0 = np.cross(t, tmp)
        norm_list.append(n0)
    # apparently, using np.vectorize is slightly faster?
    np_calc_norm = np.vectorize(calc_norm, signature='(n)->()')
    np_calc_norm(t=tangents[1:])

    norms = norm(norm_list)
    if repeat_last:
        norms = np.vstack([norms, norms[-1]])
        tangents = np.vstack([tangents, tangents[-1]])
    if not return_binormals:
        return norms
    binorms = np.cross(norms, tangents)
    return norms, binorms


def grids_to_torch(grids, size, dim=5):
    """
    convert np grids to torch.Tensor
    """
    import torch
    grids = torch.Tensor(grids / size * 2.0 - 1.0)
    while grids.dim() < dim:
        # 3d sampler is 5d, namely NCWHD
        grids = grids[None]
    return grids


def get_straight_cpr_grids(cl, angle, size=None, pivot_axis=(1, 0, 0),
                           voxel_spacing=None, sample_spacing=0.0,
                           width=40, as_torch=True):
    """
    get the sampling_grids for straight cpr
    Args:
        cl (np.ndarray): Nx3 centerline points
        angle (float or [float]): rotated angle around anchor vector,
            which is normal to rotation axis and pivot axis
        size ([w, h, d]): size of image
        pivot_axis (3d vector): Default = (1, 0, 0)
        voxel_spacing (3d vector): spacing of image
        sample_spacing (float): spacing for pixel width, as height
            spacing is predetermined by centerline spacing.
            Default = 0.0, which is sampled with unit vector
        width (int): half width around centerline. Default = 40
        as_torch (bool): Default = True. grid is scaled to [-1, 1],
            thus can be applied to torch.nn.functional.grid_sample
    Return:
        grids (np.ndarry or torch.Tensor)

    Note:
        0. If sample_spacing is given, voxel_spacing must be set
        1. If as_torch = True, size must be given.

    """
    assert (not as_torch) or (size is not None), \
        "have to set size when return as torch grid"
    assert (sample_spacing == 0.0) or (voxel_spacing is not None), \
        "have to set voxel_spacing when sample_spacing > 0.0"
    if isinstance(angle, numbers.Number):
        normals = get_consistent_normals(cl,
                                         angle=angle,
                                         pivot_axis=pivot_axis,
                                         return_binormals=False)
        sum_rep = "w,hk->hwk"

    else:
        # assuming its a series of angles
        n, bn = get_consistent_normals(cl, angle=0, return_binormals=True)
        normals = [np.cos(np.deg2rad(a)) * n +
                   np.sin(np.deg2rad(a)) * bn for a in angle]
        sum_rep = "w,chk->chwk"

    if sample_spacing > 0.0:
        norm = np.linalg.norm(normals * np.array(voxel_spacing), axis=-1)
        normals *= sample_spacing / norm[..., None]
    grids = np.einsum(sum_rep, np.arange(-width, width + 1),
                      normals) + cl[:, None]
    if as_torch:
        grids = grids_to_torch(grids, size)
    return grids


def get_stretched_cpr_grids(cl, angle, size, rot_axis=None,
                            pivot_axis=(1, 0, 0), voxel_spacing=None,
                            sample_spacing=0.0, as_torch=True,
                            return_pts2d=False):
    """
    get the sampling_grids for stretched cpr
    """
    assert (sample_spacing == 0.0) or (voxel_spacing is not None), \
        "have to set voxel_spacing when sample_spacing > 0.0"
    assert isinstance(angle, numbers.Number), "angle has to be float"

    if rot_axis is None:
        rot_axis = cl[-1] - cl[0]
    vec = get_rotated_vec(rot_axis, angle, pivot_axis)
    cl, distance = project_to_plane(cl, vec)
    r = get_range(cl, vec, size)
    end_pts = [cl + r[0] * vec, cl + r[1] * vec]
    if sample_spacing > 0.0:
        space_ratio = np.linalg.norm(vec * voxel_spacing) / sample_spacing
    else:
        space_ratio = 1.0
    width = int((r[1] - r[0])*space_ratio) + 1

    grids = np.linspace(*end_pts, width)
    if as_torch:
        grids = grids_to_torch(grids, size)
    if not return_pts2d:
        return grids
    h = (distance - r[0])*space_ratio
    pts2d = np.vstack([np.arange(len(h)), h]).T
    return grids, pts2d


def double_reflection_method(pts, r0=(1, 0, 0)):
    """
    approximation of Rotation Minimizing Frames
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf
    """
    pts = np.array(pts)
    r0 = norm(np.array(r0))
    vecs = pts[1:] - pts[:-1]
    t0 = vecs[0]
    normals = [r0]
    for index, v1 in enumerate(vecs[:-1]):
        c1 = v1.dot(v1)
        rL = r0 - (2/c1) * (v1.dot(r0)) * v1
        tL = t0 - (2/c1) * (v1.dot(t0)) * v1
        t1 = vecs[index+1]
        v2 = t1 - tL
        c2 = v2.dot(v2)
        r1 = rL - (2/c2) * (v2.dot(rL)) * v2
        normals.append(r1)
        t0 = t1
        r0 = r1

    normals = norm(normals)
    binormals = np.cross(vecs, normals)
    return normals, binormals


__all__ = [k for k in globals().keys() if not k.startswith("_")]
