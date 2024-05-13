### This is the project requirements for GSoC 2024.

### [Image Classification using Foundation Models](https://github.com/camicroscope/GSOC?tab=readme-ov-file#image-classification-using-foundation-models)

### Phase 1: Workflow Plan

1. **Foundation Model Selection:**
   - Explore the Hugging Face repository and choose a suitable foundation model for feature extraction.
   - Consider factors such as model architecture, performance on similar tasks, and computational requirements.

2. **Feature Extraction:**
   - Set up the chosen foundation model for feature extraction.
   - Define procedures to extract features from the pre-trained model for input to the classifier.
   - Implement feature extraction methods to efficiently process input data and extract relevant features.

3. **Classifier Design:**
   - Decide on the classifier architecture (e.g., MLP or CNN blocks) to be used for patch-level classification.
   - Design the architecture of the classifier, considering factors such as input size, complexity, and compatibility with the extracted features.

4. **Training Setup:**
   - Prepare the annotated patch-level dataset for training, ensuring proper data splitting for training, validation, and testing.
   - Define training parameters such as batch size, learning rate, and optimization algorithm.

5. **Training Procedure:**
   5(a) - Train only classifier
        - Train the classifier using the pre-trained features extracted from the chosen foundation model.
        - Implement training procedures to iteratively update the classifier parameters and optimize its performance on the training data.
        - Monitor training progress and evaluate performance metrics on the validation set to prevent overfitting.
   5(b) - Train/fine-tune foundation model + classifier end-to-end
        - Unfreeze the foundation model parameters
        - Train both the foundation model and classifier end-to-end 
7. **Whole Slide Image Analysis:**
   - Implement procedures to classify patches in whole slide images using the trained model.
   - Visualize and interpret the classification results to assess the model's performance on whole slide images.

8. **Documentation and Reporting:**
   - Document the entire phase, including model selection, feature extraction process, classifier architecture, training setup, evaluation results, and whole slide image classification procedure.
   - Prepare a comprehensive report summarizing the phase's objectives, methodologies, findings, and any recommendations for future work.

9. **Review and Iteration:**
   - Review the phase's outcomes and performance metrics.
   - Identify potential areas for improvement or refinement in the workflow, model architecture, or training procedures.
   - Iterate on the phase's components as necessary to enhance the overall performance and efficiency of the system.

