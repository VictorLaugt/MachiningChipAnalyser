from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from img_loader import AbstractImageLoader

import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from pathlib import Path

from img_loader import ImageDirectoryLoader, VideoFrameLoader, ImageLoadingError
from measure import measure_characteristics
from outputs_measurement_writer import MeasurementWriter
from outputs_analysis_renderer import AnalysisRenderer, NoRendering


"""Code documentation in progress:
Progression de la documentation:
[x] __main__.py
[x] img_loader.py
[x] measure.py
[x] preproc.py
[x] features_main.py
[x] features_tip.py
[x] chip_analysis.py
[ ] features_contact.py
[ ] features_thickness.py

[ ] outputs_measurement_writer.py
[ ] outputs_analysis_renderer.py
[ ] outputs_graph_animations.py

[ ] geometry.py
[ ] colors.py
[ ] type_hints.py
"""


def arg_checker_output_directory(arg: str) -> Path:
    try:
        output_directory = Path(arg)
    except ValueError:
        raise ArgumentTypeError(f"invalid path: {arg}")
    if not output_directory.parent.is_dir():
        raise ArgumentTypeError(f"invalid path: {output_directory}")
    elif output_directory.exists():
        raise ArgumentTypeError(f"file already exists: {output_directory}")
    return output_directory


def arg_checker_batch_size(arg: str) -> int:
    try:
        batch_size = int(arg)
    except ValueError:
        raise ArgumentTypeError(f"invalid size: {arg}")
    if batch_size <= 0:
        raise ArgumentTypeError(f"should be strictly positive")
    return batch_size


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description="""
The program takes a series of machining images as input and measures the following three characteristics on each of them:
- tool-chip contact length
- average peak thickness
- average valley thickness

The images taken by the input program can be supplied as a set of image files, or as a video file.
Measurement results are stored in a csv file.
In addition to make the measurements, the program can also produce graphical renderings to illustrate how the measurements are done.
"""
    )
    parser.add_argument('-S', '--silent',
        dest='no_progress_bar',
        action='store_true',
        help="if this option is enabled, the program does not display progress bar"
    )
    parser.add_argument('-i',
        dest='input_images',
        type=Path,
        required=True,
        help=(
            "path to the directory containing the images to be analyzed, or "
            "path to the video whose images are to be analyzed."
        )
    )
    parser.add_argument('-o',
        dest='output_directory',
        type=arg_checker_output_directory,
        required=True,
        help="path to the directory where output files will be written"
    )
    parser.add_argument('-s',
        dest='scale',
        type=float,
        default=1.0,
        help="length of a pixel in µm (1 by default)."
    )
    parser.add_argument('-b',
        dest='batch_size',
        type=arg_checker_batch_size,
        default=10,
        help="size of the input image batches (10 by default)."
    )
    parser.add_argument('-r',
        dest='produce_renderings',
        action='store_true',
        help=(
            "if this option is enabled, the program produces graphical renderings "
            "of the feature extractions, else, no rendering is done and the "
            "program simply extracts the features from the inputs."
        )
    )
    return parser


def build_image_loader(input_images: Path, batch_size: int) -> AbstractImageLoader:
    try:
        if input_images.is_dir():
            return ImageDirectoryLoader(input_images, ('.bmp',), batch_size)
        elif input_images.is_file():
            return VideoFrameLoader(input_images, batch_size)
        else:
            raise ValueError(f"{input_images} should be a video file or a directory containing image files")
    except ImageLoadingError:
        raise ValueError(f"unable to load input images from {input_images}")


def no_progress_bar(_iteration: int, _total: int) -> None:
    """Do not print a progress bar in the standard output"""
    return


def progress_bar(iteration: int, total: int) -> None:
    """Print a progress bar in the standard output"""
    progress = iteration / total
    bar_length = 25
    done = int(bar_length * progress)
    print(f"\r|{'#' * done}{' ' * (bar_length-done)}| {100*progress:.2f}%", end='\r')


def main():
    args = build_arg_parser().parse_args()

    # build the image loader
    try:
        loader = build_image_loader(args.input_images, args.batch_size)
    except ValueError as err:
        sys.exit(err.args[0])

    # configure the outputs
    args.output_directory.mkdir(parents=True)
    measurement_writer = MeasurementWriter(args.output_directory, args.scale)
    if args.produce_renderings:
        image_height, image_width = loader.img_shape()[:2]
        analysis_renderer = AnalysisRenderer(args.output_directory, args.scale, image_height, image_width)
    else:
        analysis_renderer = NoRendering()

    # configure the verbosity
    if args.no_progress_bar:
        progress_bar_func = no_progress_bar
        end_progress_bar_func = lambda: None
    else:
        progress_bar_func = progress_bar
        end_progress_bar_func = print

    # analyse the input images and produce the outputs
    with loader, measurement_writer, analysis_renderer:
        batch_nb = loader.batch_nb()
        for batch_idx, input_batch in enumerate(loader.img_batch_iter()):
            progress_bar_func(batch_idx, batch_nb)
            measure_characteristics(input_batch, measurement_writer, analysis_renderer)
    progress_bar_func(batch_nb, batch_nb)
    end_progress_bar_func()


if __name__ == '__main__':
    main()
