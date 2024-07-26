from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable
    from img_loader import AbstractImageLoader
    from outputs_measurement_writer import AbstractMeasurementWriter
    from outputs_analysis_renderer import AbstractAnalysisRenderer

from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from pathlib import Path

from img_loader import ImageDirectoryLoader, VideoFrameLoader
from outputs_measurement_writer import MeasurementWriter
from outputs_analysis_renderer import AnalysisRenderer, NoRendering

from preproc import preprocess
from analysis import extract_geometrical_features
from features_contact import measure_contact_length
from features_thickness import measure_spike_valley_thickness


def valid_path(value: str) -> Path:
    try:
        return Path(value)
    except ValueError:
        raise ArgumentTypeError(f"invalid path {value}")


def arg_checker_input_images(arg: str) -> AbstractImageLoader:
    input_images = valid_path(arg)
    if input_images.is_dir():
        try:
            loader = ImageDirectoryLoader(input_images, ('.bmp',))
        except FileNotFoundError:
            raise ArgumentTypeError(f"image files not found in the input directory: {input_images}")
    elif input_images.is_file():
        try:
            loader = VideoFrameLoader(input_images)
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


def no_progress_bar(_iteration: int, _total: int, _step: int) -> None:
    return

def progress_bar(iteration: int, total: int, step: int) -> None:
    if iteration % step == 0:
        progress = iteration / total
        length = 25
        done = int(length * progress)
        print(f"\r|{'#' * done}{' ' * (length-done)}| {100*progress:.2f}%", end='\r')


def analysis_loop(
    loader: AbstractImageLoader,
    measurement_writer: AbstractMeasurementWriter,
    analysis_renderer: AbstractAnalysisRenderer,
    progress: Callable[[int, int, int], None]
) -> None:
    img_nb = len(loader)
    for i, img in enumerate(loader):
        # image processing
        binary_img = preprocess(img)
        features = extract_geometrical_features(binary_img)

        # signal processing
        contact_len = measure_contact_length(features.main_ft, features.contact_ft)
        thickness_analysis = measure_spike_valley_thickness(features.main_ft, features.inside_ft)

        # output result
        measurement_writer.write(contact_len, thickness_analysis)
        analysis_renderer.render_frame(img, binary_img, features, contact_len, thickness_analysis)

        progress(i, img_nb, 10)


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
    analysis_loop(args.input_images, measurement_writer, analysis_renderer, progress_bar_func)
    measurement_writer.release()
    analysis_renderer.release()


if __name__ == '__main__':
    main()
