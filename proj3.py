import argparse
import os
import tempfile

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable
from pathlib import Path


def order_cw_from_top_left(points: np.ndarray) -> np.ndarray:
    """Re-Order a set of k points in the order tl tr br bl, by extracting
    max extent points"""
    rectangle = np.zeros((4, 2), dtype="float32")
    manhattan_distance = points.sum(axis=1)
    top_left_idx = np.argmin(manhattan_distance)
    bottom_right_idx = np.argmax(manhattan_distance)

    xy_diff = np.diff(points, axis=1)
    top_right_idx = np.argmin(xy_diff)
    bottom_left_idx = np.argmax(xy_diff)

    rectangle[[0, 1, 2, 3]] = points[[top_left_idx, top_right_idx, bottom_right_idx, bottom_left_idx]]
    return rectangle


def transform_perspective(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply perspective transform to an image give a set of 4 points."""
    rectangle = order_cw_from_top_left(points)
    (top_left, top_right, bottom_right, bottom_left) = rectangle

    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    width_top = np.linalg.norm(top_right - top_left)
    max_width = int(max(width_bottom, width_top))

    height_right = np.linalg.norm(top_right - bottom_right)
    height_left = np.linalg.norm(top_left - bottom_left)
    max_height = int(max(height_left, height_right))

    transformed_view = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    transformation_matrix = cv2.getPerspectiveTransform(rectangle, transformed_view)
    warped_image = cv2.warpPerspective(image, transformation_matrix, (max_width, max_height))
    return warped_image


def display_image_m(image: np.ndarray, **kwargs) -> None:
    """Display an image using matplotlib"""
    plt.figure()
    plt.imshow(image, **kwargs)
    plt.show()


def show_color_image(image: np.ndarray) -> None:
    """Display an RGB image using matplotlib"""
    display_image_m(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def show_gray(image: np.ndarray) -> None:
    """Display a grayscale image using matplotlib"""
    display_image_m(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')


def display_image(image: np.ndarray, title: str) -> None:
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def apply_canny_edge_detector(image: np.ndarray, dilate_and_apert=False) -> np.ndarray:
    """Converts a BGR input image to Grayscale
    and runs Canny Edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if not dilate_and_apert:
        edges = cv2.Canny(gray, 75, 200)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(gray, kernel)

        edges = cv2.Canny(dilated, 0, 84, apertureSize=3)

    return edges


def get_image_contours(image: np.ndarray) -> Iterable:
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted(contours, key=cv2.contourArea, reverse=True)


def region_of_interest(contours: Iterable) -> np.ndarray:
    max_area = -1
    screen_contour = contours[0]
    for contour in contours:
        # approximate the contour
        perimeter = cv2.arcLength(contour, False)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        bounding_rectangle = cv2.minAreaRect(contour)
        rectangle_points = cv2.boxPoints(bounding_rectangle)
        box = np.int0(rectangle_points)
        rectangle_area = cv2.contourArea(box)

        if rectangle_area > max_area:
            screen_contour = approx
            max_area = rectangle_area

    return screen_contour


def sharpen_image_text(warped_image: np.ndarray) -> np.ndarray:
    # convert the warped image to grayscale
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    # sharpen image
    sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

    # apply adaptive threshold to get black and white effect
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)

    # dilate the edges

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # dilated = cv2.dilate(thresh, kernel)

    return thresh


def perpendicular(a: np.ndarray):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perpendicular(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def cheque_detection_pipeline(image_path: str, debug: bool = False, bw_output: bool = False):
    image = cv2.imread(image_path)
    asp_ratio = image.shape[0] / image.shape[1]
    plt.rcParams["figure.figsize"] = (10, int(10 * asp_ratio))
    original = image.copy()
    resized_size = 300.0

    image = imutils.resize(image, height=int(resized_size))

    # Step 1: Edge Detection
    edges = apply_canny_edge_detector(image)

    if debug:
        show_color_image(image)

    if debug:
        print("STEP 1: Edge Detection")
        show_color_image(edges)

    # Step 2: Find contours of paper
    contours = get_image_contours(edges)
    roi = region_of_interest(contours)
    cv2.drawContours(image, [roi], -1, (0, 0, 255), 2)

    if debug:
        print("STEP 2: Find contours of paper")
        show_color_image(image)

    # Step 3: Apply ROI Cropping and Perspective Transform
    # to reshape the roi array, we get its size and dive by two
    roi_size = roi.size // 2
    ratio = original.shape[0] / resized_size
    warped = transform_perspective(original, roi.reshape(roi_size, 2) * ratio)

    if debug:
        print("STEP 3: Apply ROI Cropping and Perspective Transform")
        show_color_image(warped)

    # Step 4: Increase Contrast of The Text in Cheque
    if debug:
        print("Step 4: Increase Contrast of The Text in Cheque")

    if bw_output:
        sharpened_text = sharpen_image_text(warped)
    else:
        sharpened_text = warped

    if sharpened_text.shape[0] > sharpened_text.shape[1]:
        sharpened_text = imutils.rotate_bound(sharpened_text, 90)

    return original, sharpened_text


def extract_axis_parallel_edge(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    axis_parallel_edges = cv2.erode(image, kernel)
    return cv2.dilate(axis_parallel_edges, kernel, iterations=3)


def cheque_detection_pipeline_2(image_path: str, bw_output: bool = False):
    image = cv2.imread(image_path)
    orig = image.copy()
    resize_size = 500.0
    image = imutils.resize(image, height=int(resize_size))

    edges = apply_canny_edge_detector(image, True)

    cols = edges.shape[1]
    horizontal_size = cols // 30
    horizontal = extract_axis_parallel_edge(edges, kernel_size=(horizontal_size, 1))

    rows = edges.shape[0]
    vertical_size = rows // 30
    vertical = extract_axis_parallel_edge(edges, kernel_size=(1, vertical_size))

    edges = cv2.bitwise_or(horizontal, vertical)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 80)

    co_ord_lim = 1000000
    max_x, max_y, min_x, min_y = -co_ord_lim, -co_ord_lim, co_ord_lim, co_ord_lim
    lxmax, lymax, lxmin, lymin, = None, None, None, None

    for line in lines:
        for r, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * r
            y0 = b * r

            x1 = int(x0 + 5000 * (-b))
            y1 = int(y0 + 5000 * a)
            x2 = int(x0 - 5000 * (-b))
            y2 = int(y0 - 5000 * a)

            slope = (y2 - y1) / (x2 - x1 + 0.0000001)

            if abs(slope) > 1:
                # vertical-ish lines
                approx_xdist = abs((x2 + x1) / 2)
                if approx_xdist > max_x:
                    max_x = approx_xdist
                    lxmax = ((x1, y1), (x2, y2))

                if approx_xdist < min_x:
                    min_x = approx_xdist
                    lxmin = ((x1, y1), (x2, y2))

            else:
                approx_ydist = abs((y2 + y1) / 2)

                if approx_ydist > max_y:
                    max_y = approx_ydist
                    lymax = ((x1, y1), (x2, y2))

                if approx_ydist < min_y:
                    min_y = approx_ydist
                    lymin = ((x1, y1), (x2, y2))

    # top left point
    x1, y1 = lxmin
    x2, y2 = lymin
    top_left_pt = seg_intersect(*tuple(map(np.array, (x1, y1, x2, y2))))
    # bottom left point
    x2, y2 = lymax
    bottom_left_pt = seg_intersect(*tuple(map(np.array, (x1, y1, x2, y2))))
    # bottom right point
    x1, y1 = lxmax
    bottom_right_pt = seg_intersect(*tuple(map(np.array, (x1, y1, x2, y2))))
    # top right point
    x2, y2 = lymin
    top_right_pt = seg_intersect(*tuple(map(np.array, (x1, y1, x2, y2))))

    roi = np.array([top_left_pt, top_right_pt, bottom_right_pt, bottom_left_pt])
    roi_size = roi.size // 2
    ratio = orig.shape[0] / resize_size
    warped = transform_perspective(orig, roi.reshape(roi_size, 2) * ratio)

    if bw_output:
        warped = sharpen_image_text(warped)

    return orig, warped


def process_img(image_path: str, save_folder: str, bw_output: bool, detect_amount_folder: str):
    original, sharpened_text = cheque_detection_pipeline(image_path, bw_output=bw_output)

    w, h, _ = original.shape
    if bw_output:
        nw, nh = sharpened_text.shape
    else:
        nw, nh, _ = sharpened_text.shape

    ratio = (nw * nh) / (w * h)
    if ratio < 0.25:
        try:
            original, sharpened_text = cheque_detection_pipeline_2(image_path, bw_output=bw_output)
        except:
            return
    display_image(original, "Original")
    if save_folder:
        if not bw_output:
            image = cv2.cvtColor(sharpened_text, cv2.COLOR_BGR2RGB)
            plt.imsave(save_folder + "/" + Path(image_path).stem + ".jpg", image, dpi=300)
        else:
            image = sharpened_text
            plt.imsave(save_folder + "/" + Path(image_path).stem + ".jpg", image, dpi=300, cmap="gray")

    if detect_amount_folder:
        image = cv2.cvtColor(sharpened_text, cv2.COLOR_BGR2RGB)
        plt.imsave(detect_amount_folder + "/" + Path(image_path).stem + "cd" + ".jpg", image, dpi=300)

    display_image(sharpened_text, "Result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check preparation project")
    parser.add_argument("-i", '--input_folder', type=str, default='sample_input', help='check images folder')
    parser.add_argument("-o", "--save_to_folder", type=str, default="sample_color_output",
                        help="If specified output is saved to given folder.")

    parser.add_argument("-g", "--grayscale", action="store_true", help="Output cheques in B/W")
    parser.add_argument("-d", "--detect_amount", action="store_true", help="Try and detect the amount on the cheques")
    
    args = parser.parse_args()
    input_folder = args.input_folder
    save_folder = args.save_to_folder
    detect_amount = args.detect_amount

    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_dir = temp_dir_name if detect_amount else ""

        for check_img in os.listdir(input_folder):
            img_path = os.path.join(input_folder, check_img)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                process_img(img_path, save_folder, args.grayscale, temp_dir)

        if detect_amount:
            from src.main import infer, char_list_from_file
            from src.model import Model, DecoderType
            display_images = []
            inference_images = []
            for infile in Path(temp_dir_name).glob("*cd.jpg"):
                if infile.name.startswith("check5"):
                    continue
                image = cv2.imread(str(infile))
                width = image.shape[1]
                height = image.shape[0]
                nw = image[:(4 * height//5), (3 * width//5):, :]
                image = nw.copy()
                resized_size = 300.0
                orig = image.copy()
                ratio = orig.shape[0] / resized_size
                image = imutils.resize(image, height = int(resized_size))

                # Step 1: Edge Detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 25, 100)
                contours = get_image_contours(edges)
                max_area = -1
                screen_contour = None
                bbox = None

                for contour in contours:
                    # approximate the contour
                    perimeter = cv2.arcLength(contour, False)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                    bounding_rectangle = cv2.minAreaRect(contour)
                    rectangle_points = cv2.boxPoints(bounding_rectangle)
                    box = np.int0(rectangle_points)
                    rectangle_area = cv2.contourArea(box)
                    if len(approx) == 4:
                        # show_color(image)
                        if rectangle_area > max_area:
                            screen_contour = approx
                            bbox = contour
                            max_area = rectangle_area

                if screen_contour is not None:
                    roi = screen_contour.copy()
                    roi_size = roi.size // 2

                    warped = transform_perspective(orig.copy(), roi.reshape(roi_size, 2) * ratio)
                    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                    gray[[gray < 125]] = 0
                    gray[[gray >= 125]] = 255

                    # inference_images.append(gray)

                    # continue
                    # cv2.drawContours(orig, [bbox] * ratio, -1, (0, 255, 0), 10)
                    cv2.drawContours(image, [roi], -1, (0, 255, 0), 3)
                    org = tuple(roi[roi.squeeze().sum(axis=1).argmin()].squeeze())
                    display_images.append((image, org))

                    save_path = f"{temp_dir_name}/{Path(infile).stem}_amount.jpg"
                    plt.imsave(save_path, gray, cmap="gray", dpi=300)
                    inference_images.append(save_path)

            model = Model(char_list_from_file(), DecoderType.BestPath, must_restore=True)

            for path, dim_t in zip(inference_images, display_images):
                dim, org = dim_t
                # noinspection PyTypeChecker
                inf, prob = infer(model, path, internal_call=True)

                if "." in inf:
                    text = inf.replace(" ", "")
                else:
                    text = inf.replace(" ", ".")

                cv2.putText(dim, text, org, fontFace=cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1.0, color=(0, 0, 255), thickness=2)

                # save_img = cv2.cvtColor(dim, cv2.COLOR_BGR2RGB)
                # plt.imsave(f"check_amt_outputs/{Path(path).name}", save_img, dpi=300)

                # display the output image with text over it
                cv2.imshow("Image Text", dim)
                cv2.waitKey(0)
                cv2.destroyAllWindows()



