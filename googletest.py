import io
import os
# Imports the Google Cloud client library
# [START migration_import]
from google.cloud import vision
from google.cloud.vision import types
# [END migration_import]

def run_quickstart():
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

    #- Performs label detection on the image file ------------------------------------------------------------
    response = client.label_detection(image=image)
    labels = response.label_annotations
    for label in labels:
        print('Label: ' + label.description)
        print('   Confidence: ' + label.confidence) #maybe it is label.score

    #- Performs label detection on the image file ------------------------------------------------------------
    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    for logo in logos:
        print('Logo: ' + logo.description)
        print('   Confidence: ' + logo.confidence) #maybe it is logo.score

    #- Performs landmark detection on the image file ---------------------------------------------------------
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations

    for landmark in landmarks:
        print('Landmark: ' + landmark.description)
        for location in landmark.locations:
            lat_lng = location.lat_lng
            print('Latitude'.format(lat_lng.latitude))
            print('Longitude'.format(lat_lng.longitude))
    
    #- Performs face detection on the image file -------------------------------------------------------------
    faces = response.face_annotations

    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Faces:')

    for face in faces:
        print('Anger: {}'.format(likelihood_name[face.anger_likelihood]))
        print('Joy: {}'.format(likelihood_name[face.joy_likelihood]))
        print('Surprise: {}'.format(likelihood_name[face.surprise_likelihood]))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in face.bounding_poly.vertices])

        print('Face bounds: {}'.format(','.join(vertices)))

    #- Performs text detection on the image file -------------------------------------------------------------
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    #- Performs image property detection on the image file ---------------------------------------------------
    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    print('Properties:')

    for color in props.dominant_colors.colors:
        print('fraction: {}'.format(color.pixel_fraction))
        print('\tr: {}'.format(color.color.red))
        print('\tg: {}'.format(color.color.green))
        print('\tb: {}'.format(color.color.blue))
        print('\ta: {}'.format(color.color.alpha))

    #- Performs web entity adn page recognition on the image file --------------------------------------------
    response = client.web_detection(image=image)
    notes = response.web_detection

    if notes.pages_with_matching_images:
        print('\n{} Pages with matching images retrieved')

        for page in notes.pages_with_matching_images:
            print('Url   : {}'.format(page.url))

    if notes.full_matching_images:
        print ('\n{} Full Matches found: '.format(
               len(notes.full_matching_images)))

        for image in notes.full_matching_images:
            print('Url  : {}'.format(image.url))

    if notes.partial_matching_images:
        print ('\n{} Partial Matches found: '.format(
               len(notes.partial_matching_images)))

        for image in notes.partial_matching_images:
            print('Url  : {}'.format(image.url))

    if notes.web_entities:
        print ('\n{} Web entities found: '.format(len(notes.web_entities)))

        for entity in notes.web_entities:
            print('Score      : {}'.format(entity.score))
            print('Description: {}'.format(entity.description))

    #- Performs document text detection on the image file -----------------------------------------------------
    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    for page in document.pages:
        for block in page.blocks:
            block_words = []
            for paragraph in block.paragraphs:
                block_words.extend(paragraph.words)

            block_symbols = []
            for word in block_words:
                block_symbols.extend(word.symbols)

            block_text = ''
            for symbol in block_symbols:
                block_text = block_text + symbol.text

            print('Block Content: {}'.format(block_text))
            print('Block Bounds:\n {}'.format(block.bounding_box))
    #---------------------------------------------------------------------------------------------------------

    print('---------------------- Detection done ----------------------')


if __name__ == '__main__':
    run_quickstart()