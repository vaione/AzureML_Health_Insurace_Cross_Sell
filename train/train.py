import xgboost
import argparse
import os
import joblib
from azureml.core.run import Run
import numpy as np
import onnx
import onnxmltools
import onnxmltools.convert.common.data_types
from skl2onnx.common.data_types import FloatTensorType


run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
       
    parser.add_argument("--X_train", type=str, help="X_train")
    parser.add_argument("--X_test", type=str, help="X_test")
    parser.add_argument("--y_train", type=str, help="y_train")
    parser.add_argument("--y_test", type=str, help="y_test")
    parser.add_argument("--saved_model", type=str, help="saved_model")
    parser.add_argument("--onnx_model", type=str, help="onnx_model")
    
    parser.add_argument('--learning_rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--n_estimators', type=int, default=1000, help="Number of max estimators")
    parser.add_argument('--max_depth', type=int, default=10, help="Maximum depth of tree")
    parser.add_argument('--subsample', type=float, default=0.9, help="Subsample")
    parser.add_argument('--colsample_bytree', type=float, default=0.9, help="colsample bytree")

    args = parser.parse_args()
    run.log("Learning rate:", np.float(args.learning_rate))
    run.log("n_estimators:", np.int(args.n_estimators))
    run.log("Max depth:", np.int(args.max_depth))
    run.log("Subsample:", np.float(args.subsample))
    run.log("Colsample bytree:", np.float(args.colsample_bytree))
    
    X_train = np.loadtxt(f"{args.X_train}/X_train.csv", dtype=float)
    X_test = np.loadtxt(f"{args.X_test}/X_test.csv", dtype=float)
    y_train = np.loadtxt(f"{args.y_train}/y_train.csv", dtype=float)
    y_test = np.loadtxt(f"{args.y_test}/y_test.csv", dtype=float)

    model = xgboost.XGBClassifier(learning_rate=args.learning_rate,
                                  n_estimators=args.n_estimators,
                                  max_depth=args.max_depth,
                                  gamma=0,
                                  subsample=args.subsample,
                                  colsample_bytree=args.colsample_bytree,
                                  reg_alpha=0.1,
                                  random_state=42,
                                  tree_method='gpu_hist',
                                  gpu_id=0,
                                  predictor = 'gpu_predictor',
                                  sampling_method='gradient_based').fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    if not os.path.isdir(args.saved_model):
        os.makedirs(args.saved_model)
    if not os.path.isdir(args.onnx_model):
        os.makedirs(args.onnx_model)
    if not os.path.isdir("./outputs/model"):
        os.makedirs("./outputs/model")
    joblib.dump(model, f"{args.saved_model}/hd_model_joblib.joblib.dat")

    joblib.dump(model, "./outputs/model/hd_model_joblib.joblib.dat")
    
    num_features = 10
    initial_type = [('feature_input', FloatTensorType([1, num_features]))]

    # Convert the trained model to an ONNX format.
    onx = onnxmltools.convert.convert_xgboost(model, initial_types=initial_type)
    
    #Save onnx model
    with open(f"{args.onnx_model}/xgboost_model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

if __name__ == '__main__':
    main()


