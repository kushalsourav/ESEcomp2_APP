# import streamlit as st
# import cv2
# import numpy as np
# from mltu.inferenceModel import OnnxInferenceModel
# from mltu.utils.text_utils import ctc_decoder
# from mltu.configs import BaseModelConfigs
# import os

# st.title("Captcha Verification")

# class ImageToWordModel(OnnxInferenceModel):
#     def __init__(self, model_path, char_list, *args, **kwargs):
#         super().__init__(model_path=model_path, *args, **kwargs)
#         self.char_list = char_list

#     def predict(self, image):
#         image = cv2.resize(image, self.input_shape[:2][::-1])
#         image_pred = np.expand_dims(image, axis=0).astype(np.float32)
#         preds = self.model.run(None, {self.input_name: image_pred})[0]
#         text = ctc_decoder(preds, self.char_list)[0]
#         return text

# if __name__ == "__main__":
#     # Load model path from the code
#     model_path = "../Models/02_captcha_to_text/202401211802/model.onnx"

#     # Attempt to load model configs
#     configs_path = "../Models/02_captcha_to_text/202401211802/configs.yaml"
#     if os.path.exists(configs_path):
#         configs = BaseModelConfigs.load(configs_path)
#     else:
#         st.error(f"Configs file '{configs_path}' not found.")

#     # Create ImageToWordModel instance if configs are available
#     if 'configs' in locals():
#         model = ImageToWordModel(model_path=model_path, char_list=configs.vocab)

#         # Streamlit UI
#         uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

#         if uploaded_image is not None:
#             # Display uploaded image
#             image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
#             st.image(image, caption="Uploaded Image", use_column_width=True)

#             # Predict captcha
#             if st.button("Verify Captcha"):
#                 prediction_text = model.predict(image)
#                 st.success(f"Prediction: {prediction_text}")



import streamlit as st
import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
import os
from selenium import webdriver
from PIL import Image

cropped_path = ""  # Declare cropped_path as a global variable

st.title("Captcha Prediction")

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, model_path, char_list, *args, **kwargs):
        super().__init__(model_path=model_path, *args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text

def capture_page_screenshot(url, output_directory="captured_images", crop_params=None):
    global cropped_path  
    driver = webdriver.Chrome()
    driver.get(url)
    driver.implicitly_wait(10)

    try:
        os.makedirs(output_directory, exist_ok=True)
        screenshot_path = os.path.join(
            output_directory,
            f"captured_page_{len(os.listdir(output_directory)) + 1}.png",
        )
        driver.save_screenshot(screenshot_path)

        if crop_params:
            crop_and_save_image(screenshot_path, crop_params)
            os.remove(screenshot_path)
        else:
            print(f"Page screenshot saved to {screenshot_path}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.quit()

def crop_and_save_image(image_path, crop_params):
    global cropped_path  # Declare cropped_path as a global variable
    image = Image.open(image_path)
    cropped_image = image.crop(crop_params)
    cropped_path = image_path.replace(".png", "_cropped.png")
    cropped_image.save(cropped_path)
    print(f"Cropped image saved to {cropped_path}")

if __name__ == "__main__":
    print(cropped_path)
    model_path = "Models/02_captcha_to_text/202404031850/model.onnx"

    configs_path = "Models/02_captcha_to_text/202404031850/configs.yaml"
    if os.path.exists(configs_path):
        configs = BaseModelConfigs.load(configs_path)
    else:
        st.error(f"Configs file '{configs_path}' not found.")

    if 'configs' in locals():
        model = ImageToWordModel(model_path=model_path, char_list=configs.vocab)

        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Verify Captcha"):
                prediction_text = model.predict(image)
                st.success(f"Prediction: {prediction_text}")

        if st.button("Capture Image Automatically"):
            url = "https://kp.christuniversity.in/KnowledgePro/StudentLoginAction.do?method=studentLogoutAction"
            crop_params = (418, 468, 580, 517)
            capture_page_screenshot(url, crop_params=crop_params)
            captured_image_path = cropped_path  
            captured_image = cv2.imread(captured_image_path)
            st.image(captured_image, caption="Captured Image", use_column_width=True)
            prediction_text = model.predict(captured_image)
            st.success(f"Prediction: {prediction_text}")


