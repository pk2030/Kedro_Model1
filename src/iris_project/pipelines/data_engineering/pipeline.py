"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import evaluate_model, preprocessing, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(func=preprocessing,
             inputs="iris",
             outputs=["x","y"]),
        node(func= split_data,
             inputs =["x","y", "params:split"] ,
             outputs=["x_train", "x_test", "y_train", "y_test"])


    ])
