from distutils.command import config
from email.policy import default
from glob import glob
import os
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from core.actions import make_new_model, run_predict, run_price_predict, train_further
from core.model import Model

class UI:
    def __init__(self, configs, data):
        self.configs = configs
        self.data = data
        self.model = None

    def main_menu(self):
        while True:
            if self.model is None:
                choices = [
                    Choice("Choose a model"),
                    Choice("Train New Model"),
                    Choice("Exit")
                ]
            else:
                choices = [
                    Choice("Predict"),
                    Choice("Show Model Summary"),
                    Choice("Train Further"),
                    Choice("Exit")
                ]

            action = inquirer.select(
                message="What would you like to do?",
                choices=choices
            ).execute()
            
            match action:
                case "Choose a model":
                    model_path = inquirer.filepath("Choose a model to load", default=self.configs['model']['save_dir']).execute()
                    self.model = Model()
                    self.model.load_model(model_path)
                case "Train New Model":
                    self.model = make_new_model(self.configs, self.data)
                case "Predict":
                    run_predict(self.model, self.data, self.configs)
                case "Show Model Summary":
                    self.model.summary()
                case "Exit":
                    break
                case "Train Further":
                    train_further(self.model, self.configs, self.data)
                case _:
                    print("Invalid choice!")