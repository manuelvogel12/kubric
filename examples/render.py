# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import socket
import os
import logging
import numpy as np
import random	
import bpy
import tensorflow

import kubric as kb
from kubric.renderer import Blender


# Function to convert Cartesian coordinates to spherical coordinates
def cartesian_to_spherical(vector):
    x,y,z = vector
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


# Function to convert spherical coordinates to Cartesian coordinates
def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z


def set_camera_altitude(camera_location, angle):
    """set the altitude angle of the camera"""

    r, theta, phi = cartesian_to_spherical(camera_location)
    new_phi = np.deg2rad(90 - angle)
    camera_location_rotated = spherical_to_cartesian(r, theta, new_phi)

    return camera_location_rotated


def change_camera_angle(camera_location, angle):
    """add a certain azimuth angle to the camera"""

    r, theta, phi = cartesian_to_spherical(camera_location)
    new_theta = theta + np.deg2rad(angle)
    camera_location_rotated = spherical_to_cartesian(r, new_theta, phi)

    return camera_location_rotated


def random_camera_pos(radius):
    phi = random.uniform(0, 3.14/2.0)
    theta = random.uniform(0, 3.14 * 2.0)
    return spherical_to_cartesian(radius, theta, phi)


def check_internet(host="8.8.8.8", port=53, timeout=3):
    """
    Check internet connection by trying to connect to a known host (Google's public DNS server).
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


NUMBER_ANGLES = 12
NUMBER_RENDERS = 2
RESOLUTION = 128
OUTPUT_DIR = "output"
ALPHA_CHANNEL = True

# --- CLI arguments
parser = kb.ArgumentParser()
parser.set_defaults(
    frame_end=NUMBER_ANGLES * NUMBER_RENDERS,
    resolution=(RESOLUTION, RESOLUTION),
    job_dir=OUTPUT_DIR
)
FLAGS = parser.parse_args()


# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=32,
                   background_transparency=ALPHA_CHANNEL)

# --- Fetch shapenet
source_path = "gs://kubric-unlisted/assets/ShapeNetCore.v2.json"
shapenet = kb.AssetSource.from_manifest(source_path)

# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
scene.ambient_illumination = kb.Color(0.05, 0.05, 0.05)

# --- Get all cars
car_ids = [name for name, spec in shapenet._assets.items()
                if spec["metadata"]["category"] == "car"]


for index in range(1000, 2000):
    print(f"Current index: {index}/{len(car_ids)} ")
    asset_id = car_ids[index]
    
    # --- Keyframe the camera
    scene.camera = kb.PerspectiveCamera()

    for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
        if (frame - FLAGS.frame_start) % NUMBER_ANGLES == 0:
            # for the first frame of a render set the camera position randomly
            scene.camera.position = random_camera_pos(radius = 1.5)
        elif (frame - FLAGS.frame_start) % NUMBER_ANGLES < 6:
            # rotate by 60° horizontally, random vertical angle 
            scene.camera.position = set_camera_altitude(scene.camera.position, random.uniform(0, 75))
            scene.camera.position = change_camera_angle(scene.camera.position, 60)
        else:
            # have 6 angles with fixed 25° altitude and 60° azimuth spacing
            scene.camera.position = set_camera_altitude(scene.camera.position, 25)
            scene.camera.position = change_camera_angle(scene.camera.position, 60)
        scene.camera.look_at((0, 0, 0))
        scene.camera.keyframe_insert("position", frame)
        scene.camera.keyframe_insert("quaternion", frame)

    while True:
        try:
            while not check_internet():
                print("Waiting for internet connection...")
                time.sleep(15)

            obj = shapenet.create(asset_id=asset_id)
            logging.info(f"selected '{asset_id}'")

            # --- make object flat on X/Y and not penetrate floor
            obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
            obj.position = obj.position - (0, 0, obj.aabbox[0][2])
            scene.add(obj)
            break
        except tensorflow.python.framework.errors_impl.FailedPreconditionError:
            continue
        
    # --- Rendering
    logging.info("Rendering the scene ...")
    #renderer.save_state(output_dir / "scene.blend")
    data_stack = renderer.render()

    # --- Postprocessing
    #kb.compute_visibility(data_stack["segmentation"], scene.assets)
    #data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    #    data_stack["segmentation"],
    #    scene.assets,
    #    [obj]).astype(np.uint8)

    odir = os.path.join(output_dir, asset_id)
    kb.file_io.write_rgba_batch(data_stack["rgba"], odir)
    kb.file_io.write_depth_batch(data_stack["depth"], odir)
    # kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)

    # --- Collect metadata
    logging.info("Collecting and storing metadata for each object.")
    data = {
      "metadata": kb.get_scene_metadata(scene),
      "camera": kb.get_camera_info(scene.camera),
      "object": kb.get_instance_info(scene, [obj])
    }
    kb.file_io.write_json(filename=odir + "/z_metadata.json", data=data)
    scene.remove(obj)
    kb.done()
    
# docker run --rm --interactive --user $(id -u):$(id -g) --volume "$(pwd):/kubric" kubricdockerhub/kubruntu /usr/bin/python3 examples/render.py
