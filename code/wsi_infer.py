import subprocess
import streamlit as st

def wsi_input_streamlit():

    st.header("Whole Slide Image Inference")

    if "wsi_dir" not in st.session_state:
        st.session_state.wsi_dir = 0
    wsi_dir = st.text_input("Enter the path directory containing whole slide images:")
    st.session_state.wsi_dir = wsi_dir
    
    if "results_dir" not in st.session_state:
        st.session_state.results_dir = 0
    results_dir = st.text_input("Enter the path directory to save results: ")
    st.session_state.results_dir = results_dir

    if "model_path" not in st.session_state:
        st.session_state.model_path = 0
    model_path = st.text_input("Enter the path of the saved model: ")
    st.session_state.model_path = model_path

    if "config_path" not in st.session_state:
        st.session_state.config_path = 0
    config_path = st.text_input("Enter the path of the json configuration file: ")
    st.session_state.config_path = config_path

    return st.session_state.wsi_dir, st.session_state.results_dir, st.session_state.model_path, st.session_state.config_path

def run_wsinfer(wsi_dir, results_dir, model_path, config_path):

    """ 
    wsinfer: https://github.com/SBU-BMI/wsinfer
    wsinfer-zoo: https://github.com/SBU-BMI/wsinfer-zoo
    """
    

    command = [
        'wsinfer', 'run',
        '--wsi-dir', wsi_dir,
        '--results-dir', results_dir,
        '--model-path', model_path,
        '--config', config_path
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Output: {result.stdout}")

    

def wsi_infer():
    if 'inference_completed' not in st.session_state:
        st.session_state.inference_completed = False

    wsi_dir, results_dir, model_path, config_path = wsi_input_streamlit()
    
    if st.button("Run Inference"):
        run_wsinfer(wsi_dir, results_dir, model_path, config_path)
        st.session_state.inference_completed = True

    if st.session_state.inference_completed:
        st.write("Results saved in the directory: ", results_dir)
        st.write("Inference completed successfully!")
        st.write("Display the results:")
        # Display the results
        st.image("/home/shakib/Work/Personal/caMicroscope/Application/results/masks/TCGA-94-7943-01Z-00-DX1.361fc645-89ae-4934-8c27-12907bc2a9ee.jpg")

        if st.button("Go back to Training"):
            st.session_state.inference_completed = False
    
