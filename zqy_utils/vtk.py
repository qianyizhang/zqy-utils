
import lazy_import
import numpy as np

vtk = lazy_import.lazy_module("vtk")


def np_to_polydata(pts, cells=None, poly_type="Polys"):
    """
    convert np.points (+ faces) to vtk.polydata
    """
    polyData = vtk.vtkPolyData()
    numberOfPoints = len(pts)
    points = vtk.vtkPoints()
    for x, y, z in pts:
        points.InsertNextPoint(x, y, z)
    polyData.SetPoints(points)
    if cells is None:
        # assuming it is a line
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(numberOfPoints)
        for i in range(numberOfPoints):
            lines.InsertCellPoint(i)
        polyData.SetLines(lines)
    else:
        polys = vtk.vtkCellArray()
        for indices in cells:
            polys.InsertNextCell(len(indices))
            for ind in indices:
                polys.InsertCellPoint(ind)
        setter = getattr(polyData, f"Set{poly_type}")
        setter(polys)
    return polyData


def endpts_to_polyline(start, end, sampling_rate=1):
    """
    convert np.points to bunch of vtk.lines
    """
    if sampling_rate > 1:
        start = start[::sampling_rate]
        end = end[::sampling_rate]
    size = len(start)
    poly_pts = np.vstack([start, end])
    indices = np.vstack([np.arange(size), size+np.arange(size)]).T
    poly = np_to_polydata(poly_pts, indices, "Lines")
    return poly


def np_to_points(np_mat):
    """
    convert np.points to vtk.vtkPoints
    """
    pts = vtk.vtkPoints()
    pts.SetData(vtk.util.numpy_support.numpy_to_vtk(np_mat))
    return pts


def get_equal_length_pts(pts, sample_spacing, method="cardinal"):
    """
    given a series of points, return equal spacing sampled points
    using vtk spline to approximate the parametric curve
    """
    from vtk.util.numpy_support import vtk_to_numpy
    polyData = np_to_polydata(pts)
    spline = vtk.vtkSplineFilter()
    spline.SetInputDataObject(polyData)
    if method == "cardinal":
        spline.SetSpline(vtk.vtkCardinalSpline())
    elif method == "kochanek":
        spline.SetSpline(vtk.vtkKochanekSpline())
    else:
        pass
    spline.SetSubdivideToLength()
    spline.SetLength(sample_spacing)
    spline.Update()
    equal_length_pts = vtk_to_numpy(spline.GetOutput().GetPoints().GetData())
    return equal_length_pts


def get_parametric_pts(pts, linear=False, num_pts=-1, as_np=True):
    vtkpts = np_to_points(pts)
    spline = vtk.vtkParametricSpline()
    if linear:
        spline.SetXSpline(vtk.vtkSCurveSpline())
        spline.SetYSpline(vtk.vtkSCurveSpline())
        spline.SetZSpline(vtk.vtkSCurveSpline())
    spline.SetPoints(vtkpts)
    ret = vtk.vtkParametricFunctionSource()
    ret.SetParametricFunction(spline)
    if num_pts > 0:
        ret.SetUResolution(num_pts)
    ret.Update()
    if as_np:
        from vtk.util.numpy_support import vtk_to_numpy
        return vtk_to_numpy(ret.GetOutput().GetPoints().GetData())
    return ret


def write_mesh_to_ply(vertices, faces, filename="mesh.ply"):
    headers = ["ply", "format ascii 1.0",
               "comment author: zqy",
               f"element vertex {len(vertices)}",
               "property float x",
               "property float y",
               "property float z",
               f"element face {len(faces)}",
               "property list uchar int vertex_indices",
               "end_header", ""]
    vertex_list = [" ".join(map(str, pts)) for pts in vertices]
    face_list = [" ".join(map(str, [len(inds)] + list(inds)))
                 for inds in faces]
    with open(filename, "w") as f:
        f.write("\n".join(headers))
        f.write("\n".join(vertex_list))
        f.write("\n")
        f.write("\n".join(face_list))


def write_mesh_to_stl(poly_data, filename="mesh.stl"):
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(filename)
    stlWriter.SetInputData(poly_data)
    stlWriter.Write()


__all__ = [k for k in globals().keys() if not k.startswith("_")]
