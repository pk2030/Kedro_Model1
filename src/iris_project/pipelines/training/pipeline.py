"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.18.9
"""

# from kedro.extras.datasets.pickle import PickleDataSet


from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model,evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=train_model,
             inputs=["x_train", "y_train","params:model_params"],
             outputs="model",
             name="Train_Model"),
        node(func=evaluate_model,
             inputs=["model","x_test", "y_test"],
             outputs="confusion_matrix",
             name="Evaluate_Model")])


