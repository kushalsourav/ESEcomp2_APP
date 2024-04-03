import streamlit as st
from keras import layers
from keras.models import Model
from mltu.tensorflow.model_utils import residual_block

def about_page():
    st.title("About Captcha Verification with Custom OCR")
    st.markdown(
        """
        This Streamlit application demonstrates Captcha verification using a custom OCR (Optical Character Recognition) model. The OCR model is built using TensorFlow and the MLTU (Machine Learning Tools Utilities) library, integrating advanced features such as the Connectionist Temporal Classification (CTC) loss function to handle variable-length sequence data efficiently.

        **Key Features:**

        - **Custom OCR Model**: Utilizing TensorFlow, this application integrates a custom OCR model specifically trained for reading Captcha images.
        - **CTC Loss Function**: The OCR model employs the Connectionist Temporal Classification loss function during training, enabling accurate extraction of text from Captcha images.
        - **MLTU Library Integration**: MLTU library simplifies the process of developing and deploying machine learning models. In this project, MLTU is utilized to streamline the training process of the OCR model for Captcha text extraction.

        **How to Use:**

        1. **Upload Image**: Upload a Captcha image containing text.
        2. **Manual Verification**: Click the "Verify Captcha" button to manually verify the uploaded Captcha image.
        3. **Automatic Capture**: Alternatively, use the "Capture Image Automatically" button to capture a screenshot of a webpage with a Captcha input field, verify it automatically, and display the result.

        **Importance and Key Features:**

        Captchas (Completely Automated Public Turing tests to tell Computers and Humans Apart) are widely used in online platforms to prevent automated bots from accessing certain features or services. However, traditional Captcha systems may become vulnerable to sophisticated attacks over time. Hence, developing robust Captcha reader models is crucial for ensuring the security and integrity of online platforms.

        - **Residual Blocks**: The model architecture incorporates residual blocks, which help in training deeper networks more effectively by mitigating the vanishing gradient problem.
        - **Bidirectional LSTM (BLSTM)**: Utilizing BLSTM layers enables the model to capture contextual information from both past and future sequences, enhancing the accuracy of text recognition.
        - **Dropout Regularization**: Dropout layers are employed to prevent overfitting by randomly dropping a fraction of input units during training, thus improving generalization performance.
        - **Softmax Activation**: The final output layer employs the softmax activation function to compute the probabilities of each character in the Captcha, facilitating multi-class classification.

        **Importance of the Captcha Reader Model:**

        - **Enhanced Security**: By accurately deciphering Captchas, the model helps in preventing unauthorized access to online platforms by automated bots, thereby enhancing overall security.
        - **Improved User Experience**: Reliable Captcha recognition ensures a seamless user experience by reducing the likelihood of legitimate users being incorrectly flagged as bots.
        - **Adaptability**: The model's architecture allows for easy adaptation to different types and variations of Captchas, making it versatile and suitable for a wide range of applications.

        """
    )

if __name__ == "__main__":
    about_page()
