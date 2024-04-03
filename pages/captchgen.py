# # import streamlit as st
# # import cv2
# # import numpy as np
# # from mltu.inferenceModel import OnnxInferenceModel
# # from mltu.utils.text_utils import ctc_decoder
# # from mltu.configs import BaseModelConfigs
# # import os
# # from selenium import webdriver
# # from PIL import Image
# # import random
# # import string
# # from PIL import Image, ImageDraw, ImageFont

# # import cv2
# # import numpy as np
# # from mltu.inferenceModel import OnnxInferenceModel
# # from mltu.utils.text_utils import ctc_decoder, get_cer
# # from mltu.configs import BaseModelConfigs


# # class ImageToWordModel(OnnxInferenceModel):
# #     def __init__(self, char_list, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.char_list = char_list

# #     def predict(self, image):
# #         image = cv2.resize(image, self.input_shape[:2][::-1])
# #         image_pred = np.expand_dims(image, axis=0).astype(np.float32)
# #         preds = self.model.run(None, {self.input_name: image_pred})[0]
# #         text = ctc_decoder(preds, self.char_list)[0]
# #         return text



# # def generate_random_text(length=6):
# #     characters = string.ascii_lowercase + string.digits
# #     return ''.join(random.choice(characters) for _ in range(length))


# # def generate_image_with_text(text, crop_size=(418, 468, 580, 517), image_size=(1000, 1000), background_color=(255, 255, 255)):
# #     image = Image.new('RGB', image_size, color=background_color)
# #     cropped_image = image.crop(crop_size)
# #     draw = ImageDraw.Draw(cropped_image)
# #     font_path = "C:/Windows/Fonts/Arial.ttf"  
# #     font = ImageFont.truetype(font_path, 30)
# #     x_position = 10
# #     for char in text:
# #         text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
# #         draw.text((x_position, 10), char, fill=text_color, font=font)
# #         x_position += 25  
# #     return cropped_image

# # # def generate_and_save_captchas(model, num_captchas=500, save_directory="captchas"):
# # #     os.makedirs(save_directory, exist_ok=True)
# # #     for i in range(num_captchas):
# # #         captcha_text = generate_random_text()
# # #         captcha_image = generate_image_with_text(captcha_text)
# # #         captcha_image_path = os.path.join(save_directory, f"{captcha_text}.png")
# # #         captcha_image.save(captcha_image_path)
# # #         print(f"Generated and saved captcha {i+1}/{num_captchas} with text: {captcha_text}")




# # # Main Streamlit app
# # def main():
# #     st.title("Captcha Verification")
# #     model_path = "../Models/02_captcha_to_text/202404031850/model.onnx"

# #     configs_path = "../Models/02_captcha_to_text/202404031850/configs.yaml"
# #     if os.path.exists(configs_path):
# #         configs = BaseModelConfigs.load(configs_path)
# #     else:
# #         st.error(f"Configs file '{configs_path}' not found.")

# #     if 'configs' in locals():
# #         model = ImageToWordModel(model_path=model_path, char_list=configs.vocab)
# #         action = st.sidebar.selectbox("Select Action", ["Generate Captcha", "Verify Captcha"])
# #     # if action == "Generate Captcha":
# #     #     if st.button("Generate and Save Captchas"):
# #     #         generate_and_save_captchas(model)

# #     #     # Generate and display captcha
# #     #     captcha_text = generate_random_text()
# #     #     st.write("Generated Captcha Text:", captcha_text)
# #     #     captcha_image = generate_image_with_text(captcha_text)
# #     #     st.image(captcha_image, caption="Generated Captcha Image", use_column_width=True)


# #         if action == "Generate Captcha":
# #             # Generate and display captcha
# #             captcha_text = generate_random_text()
# #             st.write("Generated Captcha Text:", captcha_text)
# #             captcha_image = generate_image_with_text(captcha_text)
# #             st.image(captcha_image, caption="Generated Captcha Image", use_column_width=True)
# #             prediction_text = model.predict(captcha_image)
# #             st.success(f"Predicted Captcha Text: {prediction_text}")

# #         elif action == "Verify Captcha":
# #             captcha_text = generate_random_text()
# #             captcha_image = generate_image_with_text(captcha_text)
# #             st.image(captcha_image, caption="Generated Captcha Image", use_column_width=True)
# #             if st.button("Verify Captcha"):
    
# #                 prediction_text = model.predict(captcha_image)
# #                 st.success(f"Predicted Captcha Text: {prediction_text}")

# # # Run the main app
# # if __name__ == "__main__":
# #     main()



# import streamlit as st
# import cv2
# import numpy as np
# from mltu.inferenceModel import OnnxInferenceModel
# from mltu.utils.text_utils import ctc_decoder
# from mltu.configs import BaseModelConfigs
# import os
# from PIL import Image
# import random
# import string
# from PIL import Image, ImageDraw, ImageFont


# class ImageToWordModel(OnnxInferenceModel):
#     def __init__(self, char_list, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.char_list = char_list

#     def predict(self, image):
#         # Convert PIL Image to NumPy array
#         image_np = np.array(image)
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
#         image_np = cv2.resize(image_np, self.input_shape[:2][::-1])
#         image_pred = np.expand_dims(image_np, axis=0).astype(np.float32)
#         preds = self.model.run(None, {self.input_name: image_pred})[0]
#         text = ctc_decoder(preds, self.char_list)[0]
#         return text


# def generate_random_text(length=6):
#     characters = string.ascii_lowercase + string.digits
#     return ''.join(random.choice(characters) for _ in range(length))


# def generate_image_with_text(text, crop_size=(418, 468, 580, 517), image_size=(1000, 1000), background_color=(255, 255, 255)):
#     image = Image.new('RGB', image_size, color=background_color)
#     cropped_image = image.crop(crop_size)
#     draw = ImageDraw.Draw(cropped_image)
#     font_path = "C:/Windows/Fonts/Arial.ttf"  
#     font = ImageFont.truetype(font_path, 30)
#     x_position = 10
#     for char in text:
#         text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         draw.text((x_position, 10), char, fill=text_color, font=font)
#         x_position += 25  
#     return cropped_image


# # Main Streamlit app
# def main():
#     st.title("Captcha Verification")
#     model_path = "Models/02_captcha_to_text/202404031850/model.onnx"

#     configs_path = "Models/02_captcha_to_text/202404031850/configs.yaml"
#     if os.path.exists(configs_path):
#         configs = BaseModelConfigs.load(configs_path)
#     else:
#         st.error(f"Configs file '{configs_path}' not found.")

#     if 'configs' in locals():
#         model = ImageToWordModel(model_path=model_path, char_list=configs.vocab)
#         action = st.sidebar.selectbox("Select Action", ["Generate Captcha", "Verify Captcha"])

#         if action == "Generate Captcha":
#             # Generate and display captcha
#             captcha_text = generate_random_text()
#             st.write("Generated Captcha Text:", captcha_text)
#             captcha_image = generate_image_with_text(captcha_text)
#             st.image(captcha_image, caption="Generated Captcha Image", use_column_width=True)
#             prediction_text = model.predict(captcha_image)  # Convert PIL Image to NumPy array
#             st.success(f"Predicted Captcha Text: {prediction_text}")

#         elif action == "Verify Captcha":
#             captcha_text = generate_random_text()
#             captcha_image = generate_image_with_text(captcha_text)
#             st.image(captcha_image, caption="Generated Captcha Image", use_column_width=True)
#             if st.button("Verify Captcha"):
#                 prediction_text = model.predict(captcha_image)  # Convert PIL Image to NumPy array
#                 st.success(f"Predicted Captcha Text: {prediction_text}")

# # Run the main app
# if __name__ == "__main__":
#     main()


import streamlit as st
import cv2
import numpy as np
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder
from mltu.configs import BaseModelConfigs
import os
from PIL import Image
import random
import string
from PIL import Image, ImageDraw, ImageFont


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image):
        # Convert PIL Image to NumPy array
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        image_np = cv2.resize(image_np, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image_np, axis=0).astype(np.float32)
        preds = self.model.run(None, {self.input_name: image_pred})[0]
        text = ctc_decoder(preds, self.char_list)[0]
        return text


def generate_random_text(length=6):
    characters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def generate_image_with_text(text, crop_size=(418, 468, 580, 517), image_size=(1000, 1000), background_color=(255, 255, 255)):
    image = Image.new('RGB', image_size, color=background_color)
    cropped_image = image.crop(crop_size)
    draw = ImageDraw.Draw(cropped_image)
    font = ImageFont.load_default()  # Use default font
    x_position = 10
    for char in text:
        text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.text((x_position, 10), char, fill=text_color, font=font)
        x_position += 25  
    return cropped_image


# Main Streamlit app
def main():
    st.title("Captcha Verification")
    model_path = "Models/02_captcha_to_text/202404031850/model.onnx"

    configs_path = "Models/02_captcha_to_text/202404031850/configs.yaml"
    if os.path.exists(configs_path):
        configs = BaseModelConfigs.load(configs_path)
    else:
        st.error(f"Configs file '{configs_path}' not found.")

    if 'configs' in locals():
        model = ImageToWordModel(model_path=model_path, char_list=configs.vocab)
        action = st.sidebar.selectbox("Select Action", ["Generate Captcha", "Verify Captcha"])

        if action == "Generate Captcha":
            # Generate and display captcha
            captcha_text = generate_random_text()
            st.write("Generated Captcha Text:", captcha_text)
            captcha_image = generate_image_with_text(captcha_text)
            st.image(captcha_image, caption="Generated Captcha Image", use_column_width=True)
            prediction_text = model.predict(captcha_image)  # Convert PIL Image to NumPy array
            st.success(f"Predicted Captcha Text: {prediction_text}")

        elif action == "Verify Captcha":
            captcha_text = generate_random_text()
            captcha_image = generate_image_with_text(captcha_text)
            st.image(captcha_image, caption="Generated Captcha Image", use_column_width=True)
            if st.button("Verify Captcha"):
                prediction_text = model.predict(captcha_image)  # Convert PIL Image to NumPy array
                st.success(f"Predicted Captcha Text: {prediction_text}")

# Run the main app
if __name__ == "__main__":
    main()
