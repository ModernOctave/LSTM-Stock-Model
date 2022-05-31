from distutils.command import config
from email.policy import default
from glob import glob
import json
import os
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from core.actions import load_old_model, make_new_model, run_point_predict, run_price_predict, run_seq_predict, train_further
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
                    Choice("Predict Sequences"),
                    Choice("Predict Points"),
                    Choice("Predict Price"),
                    Choice("Show Model Summary"),
                    Choice("Train Further"),
                    Choice("Refresh Configs"),
                    Choice("Exit")
                ]

            action = inquirer.select(
                message="What would you like to do?",
                choices=choices
            ).execute()
            
            match action:
                case "Choose a model":
                    self.model = load_old_model(inquirer.filepath("Choose a model to load", default=self.configs['model']['save_dir']).execute())
                case "Train New Model":
                    self.model = make_new_model(self.configs, self.data)
                case "Predict Sequences":
                    run_seq_predict(self.model, self.data, self.configs)
                case "Predict Points":
                    run_point_predict(self.model, self.data, self.configs)
                case "Show Model Summary":
                    self.model.summary()
                case "Train Further":
                    train_further(self.model, self.configs, self.data)
                case "Predict Price":
                    run_price_predict(self.model, self.data, self.configs)
                case "Refresh Configs":
                    self.configs = json.load(open('config.json', 'r'))
                case "Exit":
                    break
                case _:
                    print("Invalid choice!")