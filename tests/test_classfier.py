from inference.classifier import DiSegmentPredictor


def test_load_image():
    image_path = "https://i5.walmartimages.com/asr/43995148-22bf-4836-b6d3-e8f64a73be54.5398297e6f59fc510e0111bc6ff3a02a.jpeg"
    loaded_image = DiSegmentPredictor.load_image(image_path)
    assert loaded_image.shape