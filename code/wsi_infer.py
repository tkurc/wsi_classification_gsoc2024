import os
import subprocess
import streamlit as st
from click.testing import CliRunner
from wsinfer.cli.infer import run

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

################## Using click testing (CliRunner) ##################
def run_wsinfer(wsi_dir, results_dir, model_path, config_path):
    """ 
    wsinfer: https://github.com/SBU-BMI/wsinfer
    wsinfer-zoo: https://github.com/SBU-BMI/wsinfer-zoo
    """
    runner = CliRunner()

    # Run the command
    """
    CliRunner object, which simulates running a command from the command line.
    We use runner.invoke() to run the command, passing the run function and a list of arguments.
    """
    result = runner.invoke(run, [
        '--wsi-dir', wsi_dir,
        '--results-dir', results_dir,
        '--model-path', model_path,
        '--config', config_path

    ])

    # Check the result
    if result.exit_code == 0:
        print("Command executed successfully")
    
    else:
        print(f"Command failed with exit code {result.exit_code}")
        print(result.output)


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
        if st.button("Go back to Training"):
            st.session_state.inference_completed = False