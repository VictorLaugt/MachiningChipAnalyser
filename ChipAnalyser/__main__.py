from __future__ import annotations

import argparse
from pathlib import Path

import img_loader
import analysis


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument('-p',
        dest='display_render',
        action='store_true',
        help=("# TODO: '-p' argument description")
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    if not args.output_file.is_dir():
        exit(f"Output file not found: {args.output_file}")

    if args.rendering_directory is not None and not args.rendering_directory.is_dir():
        exit(f"Rendering directory not found: {args.rendering_directory}")

    if args.input_images.is_dir():
        loader = img_loader.ImageDirectoryLoader(args.input_images, ('.bmp',))
    elif args.input_images.is_file():
        loader = img_loader.VideoFrameLoader(args.input_images)
    else:
        exit(f"Input images not found: {args.input_images}")


    analysis.analysis_loop(loader)

    if args.rendering_directory is not None and args.display_render:
        # TODO: display rendering videos
        ...


if __name__ == '__main__':
    main()
