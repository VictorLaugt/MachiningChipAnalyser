from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from img_loader import AbstractImageLoader

from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from pathlib import Path

from img_loader import ImageDirectoryLoader, VideoFrameLoader
from measure import measure_characteristics
from outputs_measurement_writer import MeasurementWriter
from outputs_analysis_renderer import AnalysisRenderer, NoRendering


def valid_path(value: str) -> Path:
    try:
        return Path(value)
    except ValueError:
        raise ArgumentTypeError(f"invalid path {value}")


def arg_checker_input_images(arg: str) -> AbstractImageLoader:
    input_images = valid_path(arg)
    if input_images.is_dir():
        try:
            loader = ImageDirectoryLoader(input_images, ('.bmp',), 10)
        except FileNotFoundError:
            raise ArgumentTypeError(f"image files not found in the input directory: {input_images}")
    elif input_images.is_file():
        try:
            loader = VideoFrameLoader(input_images, 10)
        except FileNotFoundError:
            raise ArgumentTypeError(f"wrong video file: {input_images}")
    else:
        raise ArgumentTypeError("should be a video file or a directory containing image files")
    return loader


def arg_checker_output_directory(arg: str) -> Path:
    output_directory = valid_path(arg)
    if not output_directory.parent.is_dir():
        raise ArgumentTypeError(f"invalid path: {output_directory}")
    elif output_directory.exists():
        raise ArgumentTypeError(f"file already exists: {output_directory}")
    return output_directory


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
        type=arg_checker_input_images,
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


def no_progress_bar(_iteration: int, _total: int) -> None:
    return


def progress_bar(iteration: int, total: int) -> None:
    progress = iteration / total
    bar_length = 25
    done = int(bar_length * progress)
    print(f"\r|{'#' * done}{' ' * (bar_length-done)}| {100*progress:.2f}%", end='\r')


def main():
    args = build_arg_parser().parse_args()

    # configure the outputs
    args.output_directory.mkdir(parents=True)
    measurement_writer = MeasurementWriter(args.output_directory, args.scale)
    if args.produce_renderings:
        image_height, image_width = args.input_images.img_shape()[:2]
        analysis_renderer = AnalysisRenderer(args.output_directory, args.scale, image_height, image_width)
    else:
        analysis_renderer = NoRendering()

    # configure the verbosity
    if args.no_progress_bar:
        progress_bar_func = no_progress_bar
    else:
        progress_bar_func = progress_bar

    # analyse the input images and produce the outputs
    with args.input_images, measurement_writer, analysis_renderer:
        batch_nb = args.input_images.batch_nb()
        for batch_idx, input_batch in enumerate(args.input_images.img_batch_iter()):
            progress_bar_func(batch_idx, batch_nb)
            measure_characteristics(input_batch, measurement_writer, analysis_renderer)
    progress_bar_func(batch_nb, batch_nb)
    print()


if __name__ == '__main__':
    main()
