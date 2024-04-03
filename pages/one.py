import streamlit as st
import cv2
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from mltu.configs import BaseModelConfigs
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

def main():
    st.title("Exploratory Data Analysis")

    # Load the model and data
    configs = BaseModelConfigs.load("Models/02_captcha_to_text/202404031850/configs.yaml")
    
    model = ImageToWordModel(model_path="Models/02_captcha_to_text/202404031850", char_list=configs.vocab)
    df = pd.read_csv("Models/02_captcha_to_text/202401211802/val.csv", header=None)
    print(df)
    data = df[1]
    
    st.subheader("Dataset Information")
    st.write("First 5 rows of the dataset:")
    st.write(df.head())

    st.write("\nDataset info:")
    st.write(df.info())

    st.write("\nDescription of the dataset:")
    st.write(df.describe())

    st.subheader("Distribution of String Lengths")
    lengths = [len(s) for s in data]
    fig1 = px.histogram(x=lengths, title='Distribution of String Lengths')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Number of Alphabetic Characters in Each String")
    alpha_counts = [sum(c.isalpha() for c in s) for s in data]
    fig2 = px.bar(x=range(len(data)), y=alpha_counts, title='Number of Alphabetic Characters in Each String')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Number of Numeric Characters in Each String")
    numeric_counts = [sum(c.isdigit() for c in s) for s in data]
    fig3 = px.bar(x=range(len(data)), y=numeric_counts, title='Number of Numeric Characters in Each String')
    st.plotly_chart(fig3, use_container_width=True)

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

    # Print average CER
    valid_cer = [cer for cer in accum_cer if not np.isnan(cer)]
    if valid_cer:
        st.write(f"Average CER: {np.average(valid_cer)}")
    else:
        st.write("No valid predictions were made.")

if __name__ == "__main__":
    main()
