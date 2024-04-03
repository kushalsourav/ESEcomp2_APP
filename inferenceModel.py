# import cv2
# import typing
# import numpy as np
# from mltu.inferenceModel import OnnxInferenceModel
# from mltu.utils.text_utils import ctc_decoder, get_cer

# class ImageToWordModel(OnnxInferenceModel):
#     def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.char_list = char_list

#     def predict(self, image: np.ndarray):
#         try:
#             # Check if the image is empty
#             if image is None:
#                 raise ValueError("Empty image")

#             # Resize the image
#             image = cv2.resize(image, self.input_shape[:2][::-1])

#             image_pred = np.expand_dims(image, axis=0).astype(np.float32)

#             preds = self.model.run(None, {self.input_name: image_pred})[0]

#             text = ctc_decoder(preds, self.char_list)[0]

#             return text
#         except Exception as e:
#             print(f"Error occurred during prediction: {e}")
#             return None

# if __name__ == "__main__":
#     import pandas as pd
#     from tqdm import tqdm
#     from mltu.configs import BaseModelConfigs

#     configs = BaseModelConfigs.load("Models/02_captcha_to_text/202401211802/configs.yaml")

#     model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

#     df = pd.read_csv("Models/02_captcha_to_text/202404011405/val.csv").values.tolist()

#     accum_cer = []
#     for image_path, label in tqdm(df):
#         image = cv2.imread(image_path)

#         # Check if the image is loaded successfully
#         if image is None:
#             print(f"Error loading image: {image_path}")
#             continue

#         prediction_text = model.predict(image)

#         if prediction_text is not None:
#             cer = get_cer(prediction_text, label)
#             print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
#             accum_cer.append(cer)

#     if accum_cer:
#         print(f"Average CER: {np.average(accum_cer)}")
#     else:
#         print("No valid predictions were made.")

import cv2
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
import plotly.express as px
import plotly.graph_objects as go

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        try:
            # Check if the image is empty
            if image is None:
                raise ValueError("Empty image")

            # Resize the image
            image = cv2.resize(image, self.input_shape[:2][::-1])

            image_pred = np.expand_dims(image, axis=0).astype(np.float32)

            preds = self.model.run(None, {self.input_name: image_pred})[0]

            text = ctc_decoder(preds, self.char_list)[0]

            return text
        except Exception as e:
            print(f"Error occurred during prediction: {e}")
            return None

if __name__ == "__main__":
    from mltu.configs import BaseModelConfigs

    configs = BaseModelConfigs.load("Models/02_captcha_to_text/202404031357/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    df = pd.read_csv("Models/02_captcha_to_text/202404011405/val.csv", header=None)

    # EDA
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset info:")
    print(df.info())

    print("\nDescription of the dataset:")
    print(df.describe())

    # Initialize an empty list to store CER values
    accum_cer = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = row[0]
        label = row[1]

        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        prediction_text = model.predict(image)

        if prediction_text is not None:
            cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
            accum_cer.append(cer)
        else:
            # Insert a placeholder value if no prediction was made
            accum_cer.append(np.nan)

        # Update the dataframe with predictions
        df.loc[index, 'prediction'] = prediction_text if prediction_text is not None else 'No prediction'

    # Visualization 1: Histogram of label frequencies
    fig1 = px.histogram(df, x=1, title='Histogram of Label Frequencies')

    # Visualization 2: Box plot of label lengths
    df['label_length'] = df[1].apply(len)
    print(df[1].apply(len), df[1])
    print(df['label_length'])
    fig2 = px.box(df, y='label_length', title='Box Plot of Label Lengths')

    # Visualization 3: Bar chart of label lengths
    label_lengths = df['label_length'].value_counts().sort_index()
    fig3 = px.bar(x=label_lengths.index, y=label_lengths.values, labels={'x': 'Label Length', 'y': 'Count'},
                  title='Bar Chart of Label Lengths')

    # # Visualization 4: Scatter plot of label lengths vs. CER
    # df['cer'] = accum_cer
    # fig4 = px.scatter(df, x='label_length', y='cer', title='Scatter Plot of Label Lengths vs. CER')

    # # Visualization 5: 3D scatter plot of label lengths, CER, and image paths
    # fig5 = px.scatter_3d(df, x='label_length', y='cer', z=0, title='3D Scatter Plot of Label Lengths, CER, and Image Paths')

    # Show the plots
    fig1.show()
    fig2.show()
    fig3.show()


    # Print average CER
    valid_cer = [cer for cer in accum_cer if not np.isnan(cer)]
    if valid_cer:
        print(f"Average CER: {np.average(valid_cer)}")
    else:
        print("No valid predictions were made.")
