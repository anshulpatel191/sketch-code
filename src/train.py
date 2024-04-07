from __future__ import absolute_import
from classes.model.SketchCodeModel import *
import os

DATA_INPUT_PATH = "/content/SketchToCodenew/data"
VOCAB_PATH = "/content/SketchToCodenew/vocabulary.vocab"
MODEL_OUTPUT_PATH = "/content/sketch-code/output"
VAL_SPLIT = 0.2
MAX_SEQ = 150
EPOCHS = 1



def main():
   
    data_input_path = DATA_INPUT_PATH
    validation_split = VAL_SPLIT
    epochs = EPOCHS
    model_output_path = MODEL_OUTPUT_PATH
    model_json_file = None  
    model_weights_file = None  
    augment_training_data = 1

    # Load model
    model = SketchCodeModel(model_output_path, model_json_file, model_weights_file)

    
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    # Split the datasets and save down image arrays
    training_path, validation_path = ModelUtils.prepare_data_for_training(data_input_path, validation_split, augment_training_data)

    # Begin model training
    model.train(training_path=training_path,
                validation_path=validation_path,
                epochs=epochs)

if __name__ == "__main__":
    main()
