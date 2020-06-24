

import lazy_import
vtk = lazy_import.lazy_module("vtk")
numpy_to_vtk = lazy_import.lazy_callable("vtk.util.numpy_support.numpy_to_vtk")
vtk_to_numpy = lazy_import.lazy_callable("vtk.util.numpy_support.vtk_to_numpy")


def np_to_polydata(pts, cells=None):
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
        polyData.SetPolys(polys)

    return polyData


def np_to_points(np_mat):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np_mat))
    return pts


def get_equal_length_pts(pts, sample_spacing, method="cardinal"):
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
        return vtk_to_numpy(ret.GetOutput().GetPoints().GetData())
    return ret


__all__ = [k for k in globals().keys() if not k.startswith("_")]
