from typing import Optional, Callable

import numpy
from attrs import define
from affine import Affine
from rasterio import windows
from rasterio.windows import Window
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.warp import transform_bounds
from rasterio.transform import array_bounds

import rio_tiler
from rio_tiler.types import BBox, Indexes, NoData, RIOResampling, WarpResampling
from rio_tiler.models import ImageData


class Bounds(tuple):
    """
    Tuple that holds coordinate bounds as (left, bottom, right, top)
    """
    @classmethod
    def from_array(cls, a: numpy.ndarray) -> 'Bounds':
        return cls(i.item() for i in a.ravel())

    @property
    def array(self) -> numpy.ndarray:
        return numpy.reshape(self, [2, 2])

    def clip(self, other: BBox) -> 'Bounds':
        return self.from_array(numpy.clip(self.array, other[:2], other[2:]))

    def transform_crs(self, from_crs: CRS, to_crs: CRS) -> 'Bounds':
        return type(self)(transform_bounds(from_crs, to_crs, *self))

    def area(self) -> float:
        return (self[2] - self[0]) * (self[3] - self[1])

    def ensure_y_order(self) -> 'Bounds':
        """
        Ensure that self[1] < self[3] (bottom < top)
        In cases where we have diagonal affine with positive e, rasterio might reverse the
        y bounds (i.e. return (W, N, E, S) instead of (W, S, E, N))
        (this includes `transform.array_bounds` and the 'bounds' attribue of DatasetBase)
        """
        if self[1] > self[3]:
            return type(self)((self[0], self[3], self[2], self[1]))
        return self


@define
class Extents:
    bounds: Bounds
    crs: CRS

    def to_crs(self, crs: CRS) -> 'Extents':
        return type(self)(self.bounds.transform_crs(self.crs, crs),
                          crs)

    @classmethod
    def from_dataset(cls, dataset: DatasetReader) -> 'Extents':
        return cls(Bounds(dataset.bounds).ensure_y_order(), dataset.crs)


@define
class CoordGrid:
    transform: Affine
    width: int
    height: int

    def bounds(self) -> Bounds:
        return Bounds(array_bounds(self.height, self.width, self.transform)).ensure_y_order()

    def bounds_window(self, bounds: BBox) -> Window:
        np_window = windows.from_bounds(*bounds, self.transform)
        # NOTE: windows.from_bounds() returns numpy scalar objects instead of std python numbers
        # we use `.item()` to translate back.
        # The alternative is to do the min/max operations explicitly here instead of relying
        # on rasterio implementation (as is done in `utils.get_vrt_transform`)
        return Window(*[getattr(np_window, attr).item()
                        for attr in ('col_off', 'row_off', 'width', 'height')])

    def scale(self, scale_x: float, scale_y: float) -> 'CoordGrid':
        transform = self.transform * Affine.scale(scale_x, scale_y)
        return type(self)(transform, self.width / scale_x, self.height / scale_y)

    def scaling_to(self, width: float, height: float) -> tuple[float, float]:
        return (self.width / width, self.height / height)

    def scale_to(self, width: float, height: float) -> 'CoordGrid':
        return self.scale(*self.scaling_to(width, height))


@define
class GeoGrid:
    grid: CoordGrid
    crs: CRS

    def extents(self) -> Extents:
        return Extents(self.grid.bounds(), self.crs)

    @classmethod
    def from_dict(cls, data: dict):
        grid = CoordGrid(data['transform'], data['width'], data['height'])
        return cls(grid, data['crs'])

    @classmethod
    def from_dataset(cls, dataset: DatasetReader):
        return cls.from_dict(dataset.profile)


# Window utils
def snap_to_grid(window: Window) -> Window:
    """
    Return a window with integer coordinates, that contains the input
    (non-interger) window.
    """
    return windows.Window.from_slices(*window.toslices())


def clip_to_grid(window: Window, width: int, height: int) -> Window:
    return window.intersection(Window(0, 0, width, height))


def scale_window(window: Window, scale_x: float, scale_y: float) -> Window:
    return windows.Window(window.col_off * scale_x, window.row_off * scale_y,
                          window.width * scale_x, window.height * scale_y)


# Functions used by reader
def non_empty_ratio(src_dst: DatasetReader, vrt_options: dict):
    """
    Calculate the part out of the requested WarpedVRT that actually refers to data inside
    the source dataset (the rest is nodata pixels).
    The areas are computed in crs of the source raster (I assume that vrt coords are
    translatable to src coord, but src coords might be out of bounds for conversion to
    vrt crs).
    """
    src_extents = Extents.from_dataset(src_dst)
    vrt_extents = GeoGrid.from_dict(vrt_options).extents()  # "bounds" is not in `vrt_options`

    vrt_src_bounds = vrt_extents.to_crs(src_extents.crs).bounds
    clipped_src_bounds = vrt_src_bounds.clip(src_extents.bounds)
    return clipped_src_bounds.area() / vrt_src_bounds.area()


def _patch_window_ends(window, width, height, epsilon=1e-5):
    """
    This decreases the endpoints of a window by `epsilon` if they are equal
    to the specified width/height.
    This is done in order to avoid incorrectly marking the window as "boundless"
    by the code in reader.py L172
    where I believe it should be ">" instead of ">=").
    """
    (row_start, row_stop), (col_start, col_stop) = window.toranges()
    deltax = epsilon * (col_stop == width)
    deltay = epsilon * (row_stop == height)
    return Window(window.col_off, window.row_off, window.width - deltax, window.height - deltay)


def _calc_nonempty_window(
        src_extents: Extents, vrt_geo: GeoGrid,
        width: float, height: float) -> tuple[Window, Window]:
    """
    Compute the window that refers to the non-empty part of the requested vrt
    returns:
      (`output_window`: window into the output grid (based on width & height),
       `vrt_window`: window into the vrt (possibly non-integer coordinates) )
    """
    vrt_extents = vrt_geo.extents()
    nonempty_bounds = src_extents.to_crs(vrt_extents.crs).bounds.clip(vrt_extents.bounds)

    scaling = vrt_geo.grid.scaling_to(width, height)
    output_grid = vrt_geo.grid.scale(*scaling)
    output_window = clip_to_grid(
        snap_to_grid(output_grid.bounds_window(nonempty_bounds)),
        width, height)
    vrt_window = scale_window(output_window, *scaling)
    return (output_window, vrt_window)
    # return (output_window,
    #         _patch_window_ends(vrt_window, vrt_geo.grid.width, vrt_geo.grid.height))


def clipped_read(
    src_dst: DatasetReader,
    width: int, height: int, vrt_options: dict,
    indexes: Optional[Indexes] = None,
    force_binary_mask: bool = True,
    nodata: Optional[NoData] = None,
    out_dtype: None | str | numpy.dtype = None,
    resampling_method: RIOResampling = "nearest",
    reproject_method: WarpResampling = "nearest",
    unscale: bool = False,
    post_process: Optional[Callable[[numpy.ma.MaskedArray], numpy.ma.MaskedArray]] = None,
) -> ImageData:
    vrt_grid = GeoGrid.from_dict(vrt_options)
    output_window, vrt_window = _calc_nonempty_window(
        Extents.from_dataset(src_dst), vrt_grid,
        width, height)

    sub_image_data = rio_tiler.reader.read(
        src_dst,
        indexes=indexes,
        width=output_window.width,
        height=output_window.height,
        window=vrt_window,
        nodata=nodata,
        vrt_options=vrt_options,
        out_dtype=out_dtype,
        resampling_method=resampling_method,
        reproject_method=reproject_method,
        force_binary_mask=force_binary_mask,
        unscale=unscale,
        post_process=post_process,
    )
    data = numpy.ma.masked_all(
        [sub_image_data.array.shape[0], height, width],
        dtype=sub_image_data.array.dtype)
    data[(slice(None),) + output_window.toslices()] = sub_image_data.array
    imgdata = ImageData(
        data,
        assets=sub_image_data.assets,
        bounds=vrt_grid.extents().bounds,
        crs=sub_image_data.crs,
        metadata=sub_image_data.metadata,
        band_names=sub_image_data.band_names,
        # read() only supports maximum/minimum, which do not change
        dataset_statistics=sub_image_data.dataset_statistics)
    return imgdata
