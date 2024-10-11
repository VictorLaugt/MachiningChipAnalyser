from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json

import sys
MANUAL_ANNOTATIONS_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = MANUAL_ANNOTATIONS_DIR.parent
PROGRAM_DIR = REPOSITORY_DIR.joinpath("ChipAnalyser")
sys.path.append(str(REPOSITORY_DIR))
sys.path.append(str(PROGRAM_DIR))

import numpy as np
import cv2 as cv

from ChipAnalyser.img_loader import ImageDirectoryLoader
from ChipAnalyser.preproc import preprocess
from ChipAnalyser.features_main import extract_main_features
from ChipAnalyser.features_tip import locate_tool_tip
from ChipAnalyser.chip_analysis import extract_chip_features
from ChipAnalyser.measure import pt2pt_distance, measure_peak_valley_thickness
from ChipAnalyser import colors


DARK_GREEN = (0, 127, 0)


def write_measures_header(measures_path: Path):
    with measures_path.open(mode='w') as measure_file:
        csv.writer(measure_file).writerow((
            'contact length',
            'mean peak thickness',
            'mean valley thickness'
        ))


def write_measures(measures_path: Path, contact_length, mean_peak_thk, mean_valley_thk):
    with measures_path.open(mode='a') as measure_file:
        csv.writer(measure_file).writerow((contact_length, mean_peak_thk, mean_valley_thk))


# ---- draw automatic measures

def draw_frame_number(render, frame_num):
    cv.putText(
        render,
        f"frame: {frame_num}",
        (20, render.shape[0]-20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        colors.WHITE
    )


def draw_fitting_pts(render, contact_ft):
    for kpt in contact_ft.key_pts:
        cv.circle(render, kpt, 3, colors.GREEN, thickness=-1)


def draw_contact_length(render, main_ft, tip_ft, contact_ft):
    xc, yc = contact_ft.contact_pt

    if not np.isnan(yc):
        xt, yt = tip_ft.tool_tip_pt
        # _, dx, dy = main_ft.tool_line
        dx, dy = 1, 0
        width = 50
        shift = width * (1/3)
        cv.line(render, (int(xt+width*dx), int(yt+width*dy)), (int(xt), int(yt)), colors.GREEN, 1)
        cv.line(render, (int(xc+width*dx), int(yc+width*dy)), (int(xc), int(yc)), colors.GREEN, 1)
        cv.arrowedLine(render, (int(xt+shift*dx), int(yt+shift*dy)), (int(xc+shift*dx), int(yc+shift*dy)), colors.GREEN, 2)


def draw_valley_and_peaks(render, inside_ft, thickness_analysis):
    peak_pts = inside_ft.inside_contour_pts[thickness_analysis.peak_indices]
    valley_pts = inside_ft.inside_contour_pts[thickness_analysis.valley_indices]
    for pt in peak_pts:
        cv.circle(render, pt, 3, colors.GREEN, thickness=-1)
    for pt in valley_pts:
        cv.circle(render, pt, 3, colors.GREEN, thickness=-1)


def draw_automatic_measures_legend(render):
    cv.line(render, (20, 36), (30, 36), colors.GREEN, 2)
    cv.putText(render, "automatic measures", (40, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors.WHITE)


# ---- draw labels

def draw_labels(render, label_path: Path, manual_measures_path: Path):
    def extract_shapes(data: dict, label: str) -> list:
        return [shape for shape in data["shapes"] if shape["label"] == label]

    if not label_path.is_file():
        return
    with label_path.open(mode='r') as label_file:
        label_data = json.load(label_file)

    (xt, yt), (xc, yc) = extract_shapes(label_data, "contact")[0]["points"]
    dx, dy = 1, 0
    width = 50
    shift = width * (2/3)
    cv.line(render, (int(xt+width*dx), int(yt+width*dy)), (int(xt), int(yt)), colors.RED, 1)
    cv.line(render, (int(xc+width*dx), int(yc+width*dy)), (int(xc), int(yc)), colors.RED, 1)
    cv.arrowedLine(render, (int(xt+shift*dx), int(yt+shift*dy)), (int(xc+shift*dx), int(yc+shift*dy)), colors.RED, 2)
    contact_len = np.linalg.norm((xt - xc, yt - yc))

    total_peak_thk = 0.
    peak_shapes = extract_shapes(label_data, "peak")
    for peak in peak_shapes:
        (x0, y0), (x1, y1) = peak["points"]
        cv.line(render, (int(x0), int(y0)), (int(x1), int(y1)), colors.RED, 2)
        total_peak_thk += np.linalg.norm((x0 - x1, y0 - y1))
    if len(peak_shapes) > 0:
        mean_peak_thk = total_peak_thk / len(peak_shapes)
    else:
        mean_peak_thk = np.nan

    total_valley_thk = 0.
    valley_shapes = extract_shapes(label_data, "valley")
    for valley in valley_shapes:
        (x0, y0), (x1, y1) = valley["points"]
        cv.line(render, (int(x0), int(y0)), (int(x1), int(y1)), colors.RED, 2)
        total_valley_thk += np.linalg.norm((x0 - x1, y0 - y1))
    if len(valley_shapes) > 0:
        mean_valley_thk = total_valley_thk / len(valley_shapes)
    else:
        mean_valley_thk = np.nan

    cv.line(render, (20, 16), (30, 16), colors.RED, 2)
    cv.putText(render, "manual measures", (40, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors.WHITE)

    write_measures(manual_measures_path, contact_len, mean_peak_thk, mean_valley_thk)


# ---- main loop

def copy(
    input_dir: Path,
    output_dir: Path,
    read_labels: bool
):
    manual_measures_path = output_dir.joinpath("manual_measures.csv")
    if read_labels:
        write_measures_header(manual_measures_path)

    loader = ImageDirectoryLoader(input_dir, ".bmp", 10)
    image_names = (img_path.name for img_path in loader.img_paths)
    img_num = 0

    for batch_num, batch in enumerate(loader.batch_iter(), start=1):
        print(f"batch {batch_num}/{loader.batch_nb()}", end='\r')

        for inp_img in batch:
            img_num += 1
            img_name = next(image_names)
            out_img_path = output_dir.joinpath(img_name)
            label_path = out_img_path.with_suffix('.json')
            out_img = cv.cvtColor(inp_img, cv.COLOR_GRAY2RGB)

            if read_labels:
                draw_labels(out_img, label_path, manual_measures_path)
            draw_frame_number(out_img, img_num)

            cv.imwrite(str(out_img_path), cv.cvtColor(out_img, cv.COLOR_RGB2BGR))


def copy_and_measure(
    input_dir: Path,
    output_dir: Path,
    read_labels: bool
):
    automatic_measures_path = output_dir.joinpath("automatic_measures.csv")
    manual_measures_path = output_dir.joinpath("manual_measures.csv")
    write_measures_header(automatic_measures_path)
    if read_labels:
        write_measures_header(manual_measures_path)

    loader = ImageDirectoryLoader(input_dir, ".bmp", 10)
    image_names = (inp_img_path.name for inp_img_path in loader.img_paths)
    img_num = 0

    for batch_num, input_batch in enumerate(loader.batch_iter(), start=1):
        print(f"batch {batch_num}/{loader.batch_nb()}", end='\r')

        preprocessed_batch = []
        main_features = []
        for inp_img in input_batch:
            bin_img = preprocess(inp_img)
            main_ft = extract_main_features(bin_img)

            preprocessed_batch.append(bin_img)
            main_features.append(main_ft)

        tip_ft = locate_tool_tip(preprocessed_batch, main_features)

        for inp_img, bin_img, main_ft in zip(input_batch, preprocessed_batch, main_features):
            img_num += 1
            img_name = next(image_names)
            out_img_path = output_dir.joinpath(img_name)
            label_path = out_img_path.with_suffix('.json')
            out_img = cv.cvtColor(inp_img, cv.COLOR_GRAY2RGB)

            if read_labels:
                draw_labels(out_img, label_path, manual_measures_path)

            if main_ft is None or tip_ft is None:
                draw_frame_number(out_img, img_num)
                cv.imwrite(str(out_img_path), cv.cvtColor(out_img, cv.COLOR_RGB2BGR))
                continue

            tool_penetration = pt2pt_distance(tip_ft.tool_tip_pt, main_ft.tool_base_inter_pt)
            chip_ft = extract_chip_features(bin_img, main_ft, tool_penetration)

            contact_len = pt2pt_distance(tip_ft.tool_tip_pt, chip_ft.contact_ft.contact_pt)
            thickness_analysis = measure_peak_valley_thickness(chip_ft.inside_ft.thickness, tool_penetration)

            # draw_fitting_pts(out_img, chip_ft.contact_ft)
            draw_contact_length(out_img, main_ft, tip_ft, chip_ft.contact_ft)
            draw_valley_and_peaks(out_img, chip_ft.inside_ft, thickness_analysis)
            draw_automatic_measures_legend(out_img)
            draw_frame_number(out_img, img_num)
            cv.imwrite(str(out_img_path), cv.cvtColor(out_img, cv.COLOR_RGB2BGR))

            write_measures(
                automatic_measures_path,
                contact_len,
                thickness_analysis.mean_peak_thickness,
                thickness_analysis.mean_valley_thickness
            )


if __name__ == '__main__':
    def existing_directory(arg):
        try:
            input_dir = Path(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid path: {arg}")
        if not input_dir.is_dir():
            raise argparse.ArgumentTypeError(f"directory not found: {arg}")
        return input_dir

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=existing_directory, help="directory containing the original machining images")
    parser.add_argument('-a', '--automatic', action='store_true', help="automatically measure the features and draw them on the images")
    parser.add_argument('-m', '--manual', action='store_true', help="read the label files and draw the manual measures on the images")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = MANUAL_ANNOTATIONS_DIR.joinpath(input_dir.name)

    if not output_dir.is_dir():
        output_dir.mkdir()

    if args.automatic:
        copy_and_measure(input_dir, output_dir, args.manual)
    else:
        copy(input_dir, output_dir, args.manual)
