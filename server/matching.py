import json 

def find_matches(gender, teeth):
    f = open('image_data.json')
    data = json.load(f)

    options = []

    for image in data:
        if(image.get("gender") == gender and image.get("teeth") == teeth):
            options.append(image)


