from pathlib import Path
import rasterio
import morecantile
from rio_tiler import constants, reader
from rio_tiler.io import COGReader
from rio_tiler.reader_utils import Extents

COG = Path(__file__).parent / 'fixtures' / 'test_small_cog.tiff'

tile_bbox = (3757032.814272985, 3443946.746416902, 4070118.8821290657, 3757032.814272983)


def test_my_read():
    rio = rasterio.open(COG)
    print(rio.profile)
    print(rio.width, rio.height, rio.crs)
    print(repr(rio.transform))
    tile = reader.part(
        rio, tile_bbox,
        height=512, width=512,
        dst_crs=constants.WEB_MERCATOR_CRS)
    rio.close()
    print(tile)


def test_my_tile():
    tms = morecantile.tms.get('WebMercatorQuad')
    reader = COGReader(COG)
    extents = Extents.from_dataset(reader)
    # print(extents.bounds)
    tile_bounds = list(tms.tiles(*extents.bounds, 7))
    assert len(tile_bounds) == 1
    print(list(tile_bounds[0]))
    imgdata = reader.tile(*tile_bounds[0])
    print(imgdata)
    reader.close()
