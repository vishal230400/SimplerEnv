from collections import defaultdict
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts
from transforms3d.euler import euler2axangle


class RT1Inference:
    def __init__(
        self,
        saved_model_path: str = "rt_1_x_tf_trained_for_002272480_step",
        lang_embed_model_path: str = "https://tfhub.dev/google/universal-sentence-encoder-large/5",
        image_width: int = 320,
        image_height: int = 256,
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ) -> None:
        self.lang_embed_model = hub.load(lang_embed_model_path)
        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=saved_model_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True,
        )
        self._load_inner_keras_model()

        self.image_width = image_width
        self.image_height = image_height
        self.action_scale = action_scale

        self.observation = None
        self.tfa_time_step = None
        self.policy_state = None
        self.task_description = None
        self.task_description_embedding = None

        self.policy_setup = policy_setup
        if self.policy_setup == "google_robot":
            self.unnormalize_action = False
            self.unnormalize_action_fxn = None
            self.invert_gripper_action = False
            self.action_rotation_mode = "axis_angle"
        elif self.policy_setup == "widowx_bridge":
            self.unnormalize_action = True
            self.unnormalize_action_fxn = self._unnormalize_action_widowx_bridge
            self.invert_gripper_action = True
            self.action_rotation_mode = "rpy"
        else:
            raise NotImplementedError()

    def _load_inner_keras_model(self):
        try:
            self.policy_model = self.tfa_policy._model  # access internal model
        except AttributeError:
            raise ValueError("Could not access policy_model inside SavedModelPyTFEagerPolicy.")

    def _initialize_model(self) -> None:
        self.observation = tf.nest.map_structure(
            lambda spec: tf.zeros(spec.shape, dtype=spec.dtype),
            self.tfa_policy.time_step_spec.observation
        )
        self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
        self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)
        _ = self.tfa_policy.action(self.tfa_time_step, self.policy_state)

    def _resize_image(self, image: np.ndarray | tf.Tensor) -> tf.Tensor:
        image = tf.image.resize_with_pad(image, target_width=self.image_width, target_height=self.image_height)
        image = tf.cast(image, tf.uint8)
        return image

    def _initialize_task_description(self, task_description: Optional[str] = None) -> None:
        if task_description is not None:
            self.task_description = task_description
            self.task_description_embedding = self.lang_embed_model([task_description])[0]
        else:
            self.task_description = ""
            self.task_description_embedding = tf.zeros((512,), dtype=tf.float32)

    def reset(self, task_description: str) -> None:
        self._initialize_model()
        self._initialize_task_description(task_description)

    def mc_dropout_action(
        self,
        image: np.ndarray,
        task_description: Optional[str] = None,
        mc_samples: int = 10,
    ) -> tuple[list[dict[str, np.ndarray]], dict[str, np.ndarray], dict[str, np.ndarray]]:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self.observation["image"] = image
        self.observation["natural_language_embedding"] = self.task_description_embedding
        self.tfa_time_step = ts.transition(self.observation, reward=np.zeros((), dtype=np.float32))
        policy_state = self.policy_state

        raw_actions = []
        for _ in range(mc_samples):
            policy_output = self.policy_model(
                self.tfa_time_step,
                policy_state,
                training=True  # crucial for MC dropout
            )
            raw_action = policy_output.action
            raw_actions.append({k: np.asarray(v) for k, v in raw_action.items()})

        # Compute statistics
        mean_action = {
            k: np.mean([a[k] for a in raw_actions], axis=0) for k in raw_actions[0]
        }
        std_action = {
            k: np.std([a[k] for a in raw_actions], axis=0) for k in raw_actions[0]
        }

        return raw_actions, mean_action, std_action
