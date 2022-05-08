import json

with open("../image_data.json") as f:
    data = json.load(f)


def find_matches(sex, teeth):
    options = []

    for key, image in data.items():
        if len(image.get("teeth")) == 0:
            continue
        if image.get("sex") == sex and abs(image.get("teeth")[0] - float(teeth)) < 0.1:
            options.append(key)

    return options
