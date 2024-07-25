from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from img_loader import AbstractImageLoader
    from outputs_measurement_writer import AbstractMeasurementWriter
    from outputs_feature_renderer import AbstractFeatureRenderer

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from img_loader import ImageDirectoryLoader, VideoFrameLoader
from outputs_measurement_writer import MeasurementWriter
from outputs_feature_renderer import FeatureRenderer, NoRendering

from preproc import image_preprocessing
from image_analysis import geometrical_analysis


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
    parser.add_argument('-r',
        dest='rendering_directory',
        type=Path,
        required=False,
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
    feature_renderer: AbstractFeatureRenderer
) -> None:
    for img in loader:
        binary_img = image_preprocessing(img)
        main_ft, contact_ft, inside_ft = geometrical_analysis(binary_img)

        contact_length = ...  # compute from main_ft and contact_ft
        spike_mean_thickness, valley_mean_thickness = ...  # compute from [main_ft and] inside_ft

        measurement_writer.write(contact_length, spike_mean_thickness, valley_mean_thickness)
        feature_renderer.render_frame(main_ft, contact_ft, inside_ft)


def main():
    args = build_arg_parser().parse_args()

    if not args.output_file.is_dir():
        exit(f"Output file not found: {args.output_file}")

    if args.rendering_directory is not None and not args.rendering_directory.is_dir():
        exit(f"Rendering directory not found: {args.rendering_directory}")

    if args.input_images.is_dir():
        loader = ImageDirectoryLoader(args.input_images, ('.bmp',))
    elif args.input_images.is_file():
        loader = VideoFrameLoader(args.input_images)
    else:
        exit(f"Input images not found: {args.input_images}")

    measurement_writer = MeasurementWriter(args.output_file)
    if args.rendering_directory is not None:
        h, w = loader.img_shape()
        feature_renderer = FeatureRenderer(args.rendering_directory, h, w)
    else:
        feature_renderer = NoRendering()

    analysis_loop(loader, measurement_writer, feature_renderer)
    measurement_writer.release()
    feature_renderer.release()

    if args.rendering_directory is not None and args.display_render:
        # TODO: display rendering videos
        ...


if __name__ == '__main__':
    main()
