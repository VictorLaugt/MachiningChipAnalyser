from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from img_loader import AbstractImageLoader
    from outputs_measurement_writer import AbstractMeasurementWriter
    from outputs_analysis_renderer import AbstractAnalysisRenderer

import sys
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter, ArgumentTypeError
from pathlib import Path

from img_loader import ImageDirectoryLoader, VideoFrameLoader
from outputs_measurement_writer import MeasurementWriter
from outputs_analysis_renderer import AnalysisRenderer, NoRendering
from video_player import VideoFilePlayer

from preproc import preprocess
from analysis import extract_geometrical_features
from features_contact import measure_contact_length
from features_thickness import measure_spike_valley_thickness


def valid_path(value: str) -> Path:
    try:
        return Path(value)
    except ValueError:
        raise ArgumentTypeError(f"invalid path {value}")


def arg_checker_input_images(value: str) -> AbstractImageLoader:
    input_images = valid_path(value)
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


def arg_checker_output_file(value: str) -> Path:
    output_file = valid_path(value)
    output_file_dir = output_file.parent
    if not output_file_dir.is_dir():
        raise ArgumentTypeError(f"directory not found: {output_file_dir}")
    return output_file


def arg_checker_rendering_directory(value: str) -> Path:
    rendering_directory = valid_path(value)
    if not rendering_directory.is_dir():
        raise ArgumentTypeError(f"directory not found: {rendering_directory}")
    return rendering_directory


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

    parser.add_argument('-i',
        dest='input_images',
        type=arg_checker_input_images,
        required=True,
        help=(
            "path to the directory containing the images to be analyzed, or "
            "path to the video whose images are to be analyzed"
        )
    )
    parser.add_argument('-o',
        dest='output_file',
        type=arg_checker_output_file,
        required=True,
        help=(
            "path to the csv file where each column corresponds to one of the "
            "three measured features and each row corresponds to an analyzed "
            "image."
        )
    )
    parser.add_argument('-s',
        dest='scale',
        type=float,
        default=1.0,
        help="length of a pixel in µm (1 by default)."
    )
    parser.add_argument('-r',
        dest='rendering_directory',
        type=arg_checker_rendering_directory,
        help=(
            "if given, the program produces graphical renderings of the feature "
            "extractions, in the form of video that are stored in the specified "
            "directory. If not given, no rendering is done and the program simply "
            "extracts the features from the input."
        )
    )
    parser.add_argument('-d',
        dest='display_render',
        action='store_true',
        help=("# TODO: '-d' argument description")
    )

    return parser


def analysis_loop(
    loader: AbstractImageLoader,
    measurement_writer: AbstractMeasurementWriter,
    analysis_renderer: AbstractAnalysisRenderer
) -> None:
    for img in loader:
        # image processing
        binary_img = preprocess(img)
        features = extract_geometrical_features(binary_img)

        # signal processing
        contact_len = measure_contact_length(features.main_ft, features.contact_ft)
        thickness_analysis = measure_spike_valley_thickness(features.main_ft, features.inside_ft)  # MOCK find spikes and valleys

        # output result
        measurement_writer.write(contact_len, thickness_analysis)
        analysis_renderer.render_frame(img, binary_img, features, contact_len, thickness_analysis)  # MOCK analysis rendering


def main():
    args = build_arg_parser().parse_args()

    # configure the outputs
    measurement_writer = MeasurementWriter(args.output_file, args.scale)
    if args.rendering_directory is not None:
        image_height, image_width = args.input_images.img_shape()[:2]
        contact_length_vid=args.rendering_directory.joinpath("contact-length-extraction.avi"),
        inside_contour_vid=args.rendering_directory.joinpath("inside-contour-extraction.avi"),
        thickness_graph=args.rendering_directory.joinpath("chip-thickness-evolution.avi"),
        contact_length_graph=args.rendering_directory.joinpath("contact-length-evolution.png")
        analysis_renderer = AnalysisRenderer(
            args.scale,
            image_height, image_width,
            contact_length_vid, inside_contour_vid,
            thickness_graph, contact_length_graph
        )
    else:
        analysis_renderer = NoRendering()

    # analyse the image and produces the outputs
    analysis_loop(loader, measurement_writer, analysis_renderer)
    measurement_writer.release()
    analysis_renderer.release()

    # display the outputs
    if args.rendering_directory is not None and args.display_render:
        # TODO: display renders
        ...


if __name__ == '__main__':
    main()
