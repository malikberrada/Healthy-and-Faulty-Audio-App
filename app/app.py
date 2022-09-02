import streamlit as st
import pandas as pd
import io
import base64
import librosa
import librosa.display
import numpy as np
import pickle

# configuration

st.set_page_config(
                    page_title="Healthy and faulty Audios Prediction",
                    page_icon="chart_with_upwards_trend",
                    layout="wide",
                    initial_sidebar_state="auto",
                  )


# sidebar

st.sidebar.markdown('# About Audios Prediction :')

st.sidebar.markdown("""<div style="text-align: justify;"><p>For Audio Vectorization, we used <strong>MFCC</strong> (<strong>Mel-Frequency Cepstral Coefficients</strong>) :</p><p>It's a a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency.</p></div>""", unsafe_allow_html=True)


st.sidebar.markdown("""<div style="text-align: justify;"><p>For Audio Prediction, we used <strong>SVM</strong> (<strong>Support-Vector Machines</strong>) :</p><p>They are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. Developed at AT&T Bell Laboratories, SVMs are one of the most robust prediction methods, being based on statistical learning frameworks or VC theory. VC theory is a form of computational learning theory, which attempts to explain the learning process from a statistical point of view.</p></div>""", unsafe_allow_html=True)


# title and description
st.title('Healthy and faulty Audios Prediction')
st.markdown('This application has the purpose to predict whether or not an Audio File is Healthy')
st.subheader("Informations to fill :")

model_path = r'pickle/SVM-model-Healthy-Faulty-Audios.pkl'

def features_extractor(file):
    #load the file (audio)
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    #we extract mfcc
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #in order to find out scaled feature we do mean of transpose of value
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features

file = st.file_uploader("Enter the Audio files : ", key='info_form3', accept_multiple_files=True)
info_form = st.form(key='info_form')
is_clk_pred = info_form.form_submit_button('Predict Audio')
if is_clk_pred:
    extracted_features_pred = []
    for fl in file:
        relative_path = r'Data/Test/' + fl.name
        with open(relative_path, mode='wb') as f:
            f.write(fl.getvalue())
        data = features_extractor(relative_path)
        file_name = fl.name
        extracted_features_pred.append([data, relative_path, file_name])
    pred_extracted_features_df = pd.DataFrame(extracted_features_pred, columns=['feature', 'relative_path', 'File_name'])

    X_pred = np.array(pred_extracted_features_df['feature'].tolist())

    svm_classifier = pickle.load(open(model_path, 'rb'))
    y_pred_test = svm_classifier.predict(X_pred)
    pred_extracted_features_df["Predicted_Class"] = y_pred_test
    info_form.dataframe(pred_extracted_features_df[['File_name', "Predicted_Class"]])
    towrite = io.BytesIO()
    downloaded_file = pred_extracted_features_df[['File_name', "Predicted_Class"]].to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="Wav-Audio-Files-prediction.xlsx">Download excel file</a>'
    st.markdown(linko, unsafe_allow_html=True)