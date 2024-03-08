import streamlit as st # streamlit run ./diagnosis-webapp.py
import pickle
import sklearn

LR_model = pickle.load(open('LR_model.pkl','rb'))
LDA_model = pickle.load(open('LDA_model.pkl','rb'))

# Receives the prediction of the tumor classification and returns front-end output
def classify(tumor):
    if tumor == 'B':
        return 'Benign'
    else:
        return 'Malignant'
    

def main():
    st.title("Breat Cancer Tumor Identification")

    activities = ['Logistic Regression','Linear Discriminant Analysis']

    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)

    radius_mean = st.slider('Select Average Radius of Tumor', 0.0, 30.0)
    texture_mean = st.slider('Select Average Texture of Tumor', 0.0, 40.0)
    P = st.slider('Select Perimeter of Tumor', 0.0, 200.0)
    A = st.slider('Select Area of Tumor', 0.0, 2500.0)

    inputs=[[radius_mean, texture_mean, P, A]]

    if st.button('Classify'):
        if option=='Logistic Regression':
            st.success(classify(LR_model.predict(inputs)))
        elif option=='Linear Discriminant Analysis':
            st.success(classify(LDA_model.predict(inputs)))


if __name__=='__main__':
    main()
