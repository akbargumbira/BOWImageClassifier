# coding=utf-8
import os
import pickle


def serialize_object(obj, output_path):
    """Serialize object into the specified output file."""
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_serialized_object(input_path):
    """Load serialized object from the specified path"""
    with open(input_path, "rb") as f:
        obj = pickle.load(f)
    return obj
