from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from img_loader import AbstractImageLoader
    from outputs_measurement_writer import AbstractMeasurementWriter
    from outputs_analysis_renderer import AbstractAnalysisRenderer

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from img_loader import ImageDirectoryLoader, VideoFrameLoader
from outputs_measurement_writer import MeasurementWriter
from outputs_analysis_renderer import AnalysisRenderer, NoRendering

from preproc import preprocess
from analysis import extract_geometrical_features
from features_contact import measure_contact_length
from features_thickness import measure_spike_valley_thickness


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
        type=Path,
        required=True,
        help=(
            "path to the directory containing the images to be analyzed, or "
            "path to the video whose images are to be analyzed"
        )
    )
    parser.add_argument('-o',
        dest='output_file',
        type=Path,
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
        type=Path,
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
        main_ft, contact_ft, inside_ft = extract_geometrical_features(binary_img)

        # signal processing
        contact_len = measure_contact_length(main_ft, contact_ft)
        thickness_analysis = measure_spike_valley_thickness(main_ft, inside_ft)  # MOCK find spikes and valleys

        # output result
        measurement_writer.write(contact_len, thickness_analysis)
        analysis_renderer.render_frame(main_ft, contact_ft, inside_ft, contact_len, thickness_analysis)  # MOCK analysis rendering


def main():
    args = build_arg_parser().parse_args()

    if not args.output_file.is_file():
        exit(f"Output file not found: {args.output_file}")

    if args.rendering_directory is not None and not args.rendering_directory.is_dir():
        exit(f"Rendering directory not found: {args.rendering_directory}")

    if args.input_images.is_dir():
        loader = ImageDirectoryLoader(args.input_images, ('.bmp',))
    elif args.input_images.is_file():
        loader = VideoFrameLoader(args.input_images)
    else:
        exit(f"Input images not found: {args.input_images}")

    measurement_writer = MeasurementWriter(args.output_file, args.scale)
    if args.rendering_directory is not None:
        h, w = loader.img_shape()[:2]
        analysis_renderer = AnalysisRenderer(args.rendering_directory, args.scale, h, w)
    else:
        analysis_renderer = NoRendering()

    analysis_loop(loader, measurement_writer, analysis_renderer)
    measurement_writer.release()
    analysis_renderer.release()

    if args.rendering_directory is not None and args.display_render:
        # TODO: display renders
        ...


if __name__ == '__main__':
    main()
