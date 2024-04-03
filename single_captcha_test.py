import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from mltu.configs import BaseModelConfigs


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


if __name__ == "__main__":

    configs = BaseModelConfigs.load(
        "Models/02_captcha_to_text/202401211802/configs.yaml"
    )


    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

 
    image_path = "captured_images/captured_page_1_cropped.png"
    image = cv2.imread(image_path)

    
    prediction_text = model.predict(image)


    print(f"Image: {image_path}, Prediction: {prediction_text}")