"""
Module to convert yolov8 annotation to AutoML format

"""
import argparse
import glob
import os
from pathlib import Path
from ultralytics.utils import yaml_load


def dir_path(path: str):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid path")
    

def read_yaml(cfg: str) -> list:
    """
    use utility from ultralytics to read config file to get label name.
    """
    
    if isinstance(cfg, (str, Path)):
        read_dict = yaml_load(cfg)  # load dict

        names = read_dict['names']
        classes = []

        if isinstance(names, dict):
            for key, value in names.items():
                classes.append(value)
        elif isinstance(names, list):
            for value in names:
                classes.append(value)
        
        return classes
    else:
        print('Please specify valid config file.')
        return ['object']


def convert(line: str) -> tuple[int, float, float, float, float]:
    # YOLOv4 format: 1 0.365234 0.541016 0.386719 0.847656
    # AutoML format: TRAIN,gs://cloud-ml-data/img/openimage/2851/11476419305_7b73a0128c_o.jpg,Baked goods,0.56,0.25,0.97,0.25,0.97,0.50,0.56,0.50
    class_no, center_x, center_y, width, height = line.split(' ')

    class_no = int(class_no)

    center_x = float(center_x)
    center_y = float(center_y)
    width = float(width)
    height = float(height)

    xmin = center_x - width/2
    ymin = center_y - height/2
    xmax = xmin + width
    ymax = ymin + height

    return (class_no, xmin, ymin, xmax, ymax)


def convert2createml(line: str) -> tuple[int, float, float, float, float]:
    # YOLOv4 format: 1 0.365234 0.541016 0.386719 0.847656
    class_no, center_x, center_y, width, height = line.split(' ')

    class_no = int(class_no)

    center_x = float(center_x)
    center_y = float(center_y)
    width = float(width)
    height = float(height)

    return (class_no, center_x, center_y, width, height)


def export2csv(output_lines: list, filename: str):
    with open(filename, 'w') as f:
        for line in output_lines:
            f.write(line)
            f.write('\n')


def yolo2automl(
    gs_path: str = 'gs://dataset_640/',
    yolo_txts_path: str = 'datasets/dataset640/output',
    config_file: str = 'datasets/dataset640/data.yaml',
    output: str = 'output.csv',
    ):

    # only to make sure gs path ends with /
    gs_path = gs_path if gs_path.endswith('/') else f'{gs_path}/'

    # get classes from config yaml
    classes = read_yaml(config_file)

    output_lines = []  # output line for writing to csv

    # read all annotation files in path
    file_list = sorted(glob.glob(os.path.join(yolo_txts_path, '*.txt')))

    # Loop through file list
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.read().split('\n')
            for line in lines:
                if len(line) > 1:
                    # convert each line to class number, x_min, y_min, x_max, y_max
                    class_no, xmin, ymin, xmax, ymax = convert(line)

                    # final should be "UNASSIGNED,gs://dataset/cars/images/0001.jpg,class name,xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax""
                    line_temp = f'UNASSIGNED,{gs_path}{os.path.basename(file.removesuffix(".txt"))}.jpg,{classes[class_no]},{xmin},{ymin},,,{xmax},{ymax},,'
                    
                    # save converted result
                    output_lines.append(line_temp)


    try:
        export2csv(output_lines, output)
    except BaseException as e:
        print('Failed to export to csv:', e)
        exit(1)


def yolo2createml(
    yolo_txts_path: str = 'datasets/PID_output',
    config_file: str = 'datasets/yaml/data_v8s.yaml',
    output_file: str = 'output.json',
    ):

    import json

        # get classes from config yaml
    classes = read_yaml(config_file)

    # read all annotation files in path
    file_list = sorted(glob.glob(os.path.join(yolo_txts_path, '*.txt')))

    output_json = []

    # Loop through file list
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.read().split('\n')
            imagefilename = os.path.basename(file).replace('.txt', '.jpg')
            annotations = []
            for line in lines:
                if len(line) > 1:
                    # convert each line to class number, x_min, y_min, x_max, y_max
                    class_no, center_x, center_y, width, height = convert(line)
                    coordinates = {
                        "y": float(center_y),
                        "x": float(center_x),
                        "height": float(height),
                        "width": float(width)
                    }
                    label = classes[int(class_no)]
                    annotations.append({
                        "coordinates": coordinates,
                        "label": label
                    })
        output_json.append({
            "image": imagefilename,
            "annotation": annotations
        })

    with open(output_file, 'w') as fp:
        json.dump(output_json, fp)


def convert_pdf_to_jpg(pdf_file_path: str, output_folder: str, max_width: int = 4961, resolution: int = 300):
    """
    convert pdf_file_path to jpg and store in output_folder with max_width dimension and resolution.
    """

    if len(pdf_file_path) < 1:
        return
    
    import fitz

    # Open the PDF document
    pdf_document = fitz.open(pdf_file_path) # type: ignore

    # Extract file name
    base_filename = os.path.splitext(os.path.basename(pdf_file_path))[0]

    # Convert each page to JPEG images with specific resolution
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        image = page.get_pixmap(matrix=fitz.Matrix(1, 1).prescale(resolution / 72, resolution / 72))

        # If current image width larger than max_width charge the size of converted image
        width, height = image.width, image.height

        if width > max_width:
            new_height = int(height * max_width / width)
            image = page.get_pixmap(matrix=fitz.Matrix(1, 1).prescale(resolution / 72, resolution / 72))

        # Generate image file name to save
        image_file_poth = os.path.join(output_folder, f"{base_filename}_page{page_number + 1}.jpg")

        # Save image
        image.save(image_file_poth)

    # Close the PDF document
    pdf_document.close()


def batch_convert_pdf_to_jpg(input_folder: str, output_folder: str, max_width: int = 4961, resolution: int = 300):
    """
    """
    # Create output folder if not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Searching for all PDF files in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(input_folder, filename)

            # Call convert function for each file
            convert_pdf_to_jpg(pdf_file_path, output_folder, max_width, resolution)

            # log for convert file
            print(f"Converted {filename} to JPG images.")


if __name__ == '__main__':
    yolo2createml()
