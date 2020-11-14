

import os
import pandas as pd
import xml.etree.ElementTree as ET

from glob import glob

def xml_to_csv(xml_folder, output_file=None):
    """Converts a folder of XML label files into a pandas DataFrame and/or
    CSV file, which can then be used to create a :class:`detecto.core.Dataset`
    object. Each XML file should correspond to an image and contain the image
    name, image size, image_id and the names and bounding boxes of the objects in the
    image, if any. Extraneous data in the XML files will simply be ignored.
    See :download:`here <../_static/example.xml>` for an example XML file.
    For an image labeling tool that produces XML files in this format,
    see `LabelImg <https://github.com/tzutalin/labelImg>`_.
    :param xml_folder: The path to the folder containing the XML files.
    :type xml_folder: str
    :param output_file: (Optional) If given, saves a CSV file containing
        the XML data in the file output_file. If None, does not save to
        any file. Defaults to None.
    :type output_file: str or None
    :return: A pandas DataFrame containing the XML data.
    :rtype: pandas.DataFrame
    **Example**::
        >>> from detecto.utils import xml_to_csv
        >>> # Saves data to a file called labels.csv
        >>> xml_to_csv('xml_labels/', 'labels.csv')
        >>> # Returns a pandas DataFrame of the data
        >>> df = xml_to_csv('xml_labels/')
    """

    xml_list = []
    image_id = 0
    # Loop through every XML file
    for xml_file in glob(xml_folder + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Each object represents each actual image label
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (filename, width, height, label, int(float(box.find('xmin').text)),
                   int(float(box.find('ymin').text)), int(float(box.find('xmax').text)), int(float(box.find('ymax').text)), image_id)
            xml_list.append(row)
        
        image_id += 1

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
    xml_df = pd.DataFrame(xml_list, columns=column_names)

    if output_file is not None:
        xml_df.to_csv(output_file, index=None)

    return xml_df

xml_to_csv('dhaka/val_voc/','dhaka/val_xml.csv')