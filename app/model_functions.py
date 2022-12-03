from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


class CNNModel:

    def __init__(self, filepath: str, test_image_path: str):
        self.model = load_model(filepath=filepath)
        self.predict_image(image_path=test_image_path)

    def __prepare_image__(self, filename: str):
        img = load_img(filename, target_size=(224, 224))
        img = img_to_array(img)
        img = img.reshape(1, 224, 224, 3)
        img = img.astype('float32')
        img = img - [123.68, 116.779, 103.939]
        return img

    def predict_image(self, image_path: str):
        img = self.__prepare_image__(filename=image_path)
        result = self.model.predict(img)
        return result[0]
