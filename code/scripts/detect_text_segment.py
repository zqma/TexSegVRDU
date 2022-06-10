from glob import glob
from textseg.predictor import MyPredictor
from textseg.utils import read_config, read_im

def predict(image_folder: str, batch_size):
    proj_config = read_config()
    predictor = MyPredictor()
    images = []
    image_paths = list(glob(f"{image_folder}/*.png"))
    for i in range(0, len(image_paths), batch_size):
        for image_path in image_paths[i: i + batch_size]:
            images.append(read_im(image_path))
        outputs = predictor(images)




