# run_training.py
import torch
from src.load_data import load_data
from src.embedding_model import IterativeEmbeddingModel
from src.train_model import train_model_across_graphs
from src.save_parameters import save_model_parameters

file_path = '/Users/prishajain/Desktop/self_supervised_mis/dataset_buffer/collab_graphs.pickle'
graphs = load_data(file_path)
model = IterativeEmbeddingModel(p=8)
train_model_across_graphs(model, graphs)
save_model_parameters(model)
