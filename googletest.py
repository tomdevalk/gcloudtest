# [START vision_quickstart]
import io
import os
import google
from google import cloud
# Imports the Google Cloud client library
# [START migration_import]
from google.cloud import vision
from google.cloud.vision import types
def run_quickstart():

    # [END migration_import]

    # Instantiates a client
    # [START migration_client]
    client = vision.ImageAnnotatorClient()
    # [END migration_client]

    # The name of the image file to annotate
    file_name = os.path.join(
        os.path.dirname(__file__),
        'bk-chicken-fries-hed-2014.jpg')

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.logo_detection(image=image)
    logos = response.logo_annotations

    for logo in logos:
        print('Logo: ' + logo.description)
        print('   Confidence: ' + logo.confidence)
    print('Detection done')
    # [END vision_quickstart]


if __name__ == '__main__':
    run_quickstart()