#!/usr/bin/env python3
# Copyright 2021 by Julian Nubert, Robotic Systems Lab, ETH Zurich.
# All rights reserved.
# This file is released under the "BSD-3-Clause License".
# Please see the LICENSE file that has been included as part of this package.
import time

import mlflow
import mlflow.pytorch
import pickle
import torch
import numpy as np
import qqdm

import deploy.deployer


class Trainer(deploy.deployer.Deployer):

    def __init__(self, config):
        super(Trainer, self).__init__(config=config)
        self.training_bool = True
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.config["learning_rate"])

        # Load checkpoint
        if self.config["checkpoint"]:
            checkpoint = torch.load(self.config["checkpoint"], map_location=self.device)
            ## Model weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print('\033[92m' + "Model weights loaded from " + self.config["checkpoint"] + "\033[0;0m")
            ## Optimizer parameters
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print('\033[92m' + "Optimizer parameters loaded from " + self.config["checkpoint"] + "\033[0;0m")
            ## Directly train unsupervised in that case, since model is pretrained
            self.config["unsupervised_at_start"] = True

        if self.config["inference_only"]:
            print(
                "Config error: Inference only does not make sense during training. Changing to inference_only=False.")
            self.config["inference_only"] = False

    def train_epoch(self, epoch, dataloader):

        epoch_losses = {
            "loss_epoch": 0.0,
            "loss_point_cloud_epoch": 0.0,
            "loss_field_of_view_epoch": 0.0,
            "loss_po2po_epoch": 0.0,
            "loss_po2pl_epoch": 0.0,
            "loss_pl2pl_epoch": 0.0,
            "visible_pixels_epoch": 0.0,
            "loss_yaw_pitch_roll_epoch": np.zeros(3),
            "loss_true_trafo_epoch": 0.0,
        }
        counter = 0

        qqdm_dataloader = qqdm.qqdm(dataloader, desc=qqdm.format_str('blue', 'Epoch ' + str(epoch)))

        for preprocessed_dicts in qqdm_dataloader:
            # Load corresponnding preprocessed kd_tree
            for preprocessed_dict in preprocessed_dicts:
                # Move data to devices:
                for key in preprocessed_dict:
                    if hasattr(preprocessed_dict[key], 'to'):
                        preprocessed_dict[key] = preprocessed_dict[key].to(self.device)

            self.optimizer.zero_grad()

            epoch_losses, _ = (
                self.step(
                    preprocessed_dicts=preprocessed_dicts,
                    epoch_losses=epoch_losses,
                    log_images_bool=counter == self.steps_per_epoch - 1 or counter == 0))

            # Plotting and logging --> only first one in batch
            preprocessed_data = preprocessed_dicts[0]
            # Plot at very beginning to see initial state of the network
            if epoch == 0 and counter == 0 and not self.config["po2po_alone"]:
                self.log_image(epoch=epoch, string="_start" + "_" + preprocessed_data["dataset"])

            qqdm_dataloader.set_infos({'loss': f'{float(epoch_losses["loss_epoch"] / (counter + 1)):.6f}',
                                       'loss_point_cloud': f'{float(epoch_losses["loss_point_cloud_epoch"] / (counter + 1)):.6f}',
                                       'loss_po2po': f'{float(epoch_losses["loss_po2po_epoch"] / (counter + 1)):.6f}',
                                       'loss_po2pl': f'{float(epoch_losses["loss_po2pl_epoch"] / (counter + 1)):.6f}',
                                       'loss_pl2pl': f'{float(epoch_losses["loss_pl2pl_epoch"] / (counter + 1)):.6f}',
                                       'visible_pixels': f'{float(epoch_losses["visible_pixels_epoch"] / (counter + 1)):.6f}'})

            counter += 1

        return epoch_losses

    def train(self):

        dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 collate_fn=Trainer.list_collate,
                                                 num_workers=self.config["num_dataloader_workers"],
                                                 pin_memory=True if self.config[
                                                                        "device"] == torch.device("cuda") else False)

        # Check whether experiment already exists
        client = mlflow.tracking.MlflowClient()
        experiment_list = client.list_experiments()
        id = None
        for experiment in experiment_list:
            if experiment.name == self.config["experiment"]:
                id = experiment.experiment_id

        if id is None:
            print("Creating new MLFlow experiment: " + self.config["experiment"])
            id = mlflow.create_experiment(self.config["experiment"])
        else:
            print("MLFlow experiment " + self.config["experiment"] + " already exists. Starting a new run within it.")
        print("----------------------------------")

        with mlflow.start_run(experiment_id=id, run_name="Training: " + self.config["training_run_name"]):
            self.log_config()
            for epoch in range(10000):
                # Train for 1 epoch
                epoch_losses = self.train_epoch(epoch=epoch, dataloader=dataloader)

                # Compute metrics
                epoch_losses["loss_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_point_cloud_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_po2po_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_po2pl_epoch"] /= self.steps_per_epoch
                epoch_losses["loss_pl2pl_epoch"] /= self.steps_per_epoch
                epoch_losses["visible_pixels_epoch"] /= self.steps_per_epoch

                # Print update
                print("--------------------------")
                print("Epoch Summary: " + format(epoch, '05d') + ", loss: " + str(
                    epoch_losses["loss_epoch"]) + ", unsupervised: " + str(
                    self.config["unsupervised_at_start"]))

                # Logging
                print("Logging metrics and artifacts...")
                # Log metrics
                mlflow.log_metric("loss", float(epoch_losses["loss_epoch"]), step=epoch)
                mlflow.log_metric("loss point cloud", float(epoch_losses["loss_point_cloud_epoch"]),
                                  step=epoch)
                mlflow.log_metric("loss po2po", float(epoch_losses["loss_po2po_epoch"]),
                                  step=epoch)
                mlflow.log_metric("loss po2pl", float(epoch_losses["loss_po2pl_epoch"]),
                                  step=epoch)
                mlflow.log_metric("loss pl2pl", float(epoch_losses["loss_pl2pl_epoch"]),
                                  step=epoch)
                mlflow.log_metric("visible pixels", float(epoch_losses["visible_pixels_epoch"]),
                                  step=epoch)

                # Save latest checkpoint, and create checkpoint backup all 5 epochs
                ## Every epoch --> will always be overwritten by latest version
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': float(epoch_losses["loss_epoch"]),
                    'parameters': self.config
                }, "/tmp/" + self.config["training_run_name"] + "_latest_checkpoint.pth")
                mlflow.log_artifact("/tmp/" + self.config["training_run_name"] + "_latest_checkpoint.pth")
                ## All 5 epochs --> will be logged permanently in MLFlow
                if not epoch % 5:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': float(epoch_losses["loss_epoch"]),
                        'parameters': self.config
                    }, "/tmp/" + self.config["training_run_name"] + "_checkpoint_epoch_" + str(epoch) + ".pth")
                    mlflow.log_artifact(
                        "/tmp/" + self.config["training_run_name"] + "_checkpoint_epoch_" + str(epoch) + ".pth")

                # Save latest pickled full model
                if not self.config["use_jit"]:
                    mlflow.pytorch.log_model(self.model, "latest_model_pickled")

                if self.config["visualize_images"] and not self.config["po2po_alone"]:
                    self.log_image(epoch=epoch, string="_image")

                print("...done.")

                if not self.config["unsupervised_at_start"] and epoch_losses["loss_epoch"] < 1e-2:
                    self.config["unsupervised_at_start"] = True
                    print("Loss has decreased sufficiently. Switching to unsupervised mode.")
