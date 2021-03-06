from typing import List

import timm
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.nn_model import NNModule
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from .resnet import resnet50
from .vit import ViT

def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0]
    return predictions.view(num_batches, -1, Trajectory.state_size())


class RasterModel(NNModule):

    def __init__(self,
                 feature_builders: List[AbstractFeatureBuilder],
                 target_builders: List[AbstractTargetBuilder],
                 model_name: str,
                 attention: str,
                 pretrained: bool,
                 num_input_channels: int,
                 num_features_per_pose: int,
                 future_trajectory_sampling: TrajectorySampling
                 ):
        """
        Wrapper around raster-based CNN model
        :param feature_builders: list of builders for features
        :param target_builders: list of builders for targets
        :param model_name: name of the model (e.g. resnet_50, efficientnet_b3)
        :param pretrained: whether the model will be pretrained
        :param num_input_channels: number of input channel of the raster model.
        :param num_features_per_pose: number of features per single pose
        :param future_trajectory_sampling: parameters of predicted trajectory
        """
        super().__init__(feature_builders=feature_builders, target_builders=target_builders,
                         future_trajectory_sampling=future_trajectory_sampling)

        num_output_features = future_trajectory_sampling.num_poses * num_features_per_pose
        timm.list_models('resnet*')
        # model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=num_input_channels)
        # adding attention
        if model_name == 'resnet50':
            assert attention in ['CBAM', 'SE', 'ACM', 'NL', '']
            print('using attention {}'.format(attention))
            self._model = resnet50(pretrained=pretrained, att_type=attention)
            self._model.conv1 = torch.nn.Conv2d(num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self._model.num_features = 2048
        elif model_name == 'vit':
            print('using model VIT')
            self._model = ViT(
                image_size=256,
                channels=4,
                patch_size=32,
                num_classes=num_output_features,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
        elif model_name == 'vit2':
            print('using model VIT')
            self._model = ViT(
                image_size=256,
                channels=4,
                patch_size=16,
                num_classes=num_output_features,
                dim=1024,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
        else:
            raise ValueError('not implemented yet')


        if hasattr(self._model, 'classifier'):
            mlp = torch.nn.Linear(in_features=self._model.num_features, out_features=num_output_features)
            self._model.classifier = mlp
        elif hasattr(self._model, 'fc'):
            mlp = torch.nn.Linear(in_features=self._model.num_features, out_features=num_output_features)
            self._model.fc = mlp
        elif 'vit' in model_name:
            pass
        else:
            raise NameError('Expected output layer named "classifier" or "fc" in model')

        # img = torch.randn(1, 4, 256, 256)
        # preds = self._model(img)

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "raster": Raster,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        raster: Raster = features["raster"]

        (predictions, future_frames) = self._model.forward(raster.data)

        return {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions)), "future_frames": future_frames}
