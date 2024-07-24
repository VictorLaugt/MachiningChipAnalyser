from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from img_loader import AbstractImageLoader
    from chip_extract import MainFeatures
    from contact_measurement import ContactFeatures
    from thickness_measurement import InsideFeatures
    from measurement_writer import AbstractMeasurementWriter
    from feature_renderer import AbstractFeatureRenderer


from preproc import image_preprocessing
from shape_detect import geometrical_analysis


def analysis_loop(
    loader: AbstractImageLoader,
    measurement_writer: AbstractFeatureRenderer,
    feature_renderer: AbstractFeatureRenderer
) -> None:
    for img in loader:
        binary_img = image_preprocessing(img)
        main_ft, contact_ft, inside_ft = geometrical_analysis(binary_img)

        contact_length = ...
        spike_mean_thickness = ...
        valley_mean_thickness = ...
        measurement_writer.write(contact_length, spike_mean_thickness, valley_mean_thickness)

        feature_renderer.render_frame(main_ft, contact_ft, inside_ft)
