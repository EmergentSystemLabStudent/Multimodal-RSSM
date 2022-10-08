import torch
import numpy as np

import copy


def shift_next_time(pose):
  if torch.is_tensor(pose):
    pose_next = torch.zeros_like(pose)
  else:
    pose_next = np.zeros_like(pose)
  pose_next[:-1] = pose[1:]
  return pose_next

def shift_prev_time(pose):
  if torch.is_tensor(pose):
    pose_prev = torch.zeros_like(pose)
  else:
    pose_prev = np.zeros_like(pose)
  pose_prev[:-1] = pose[1:]
  return pose_prev


# v1
def normalize_pose_rpy_v1(pose_rpy):
  if torch.is_tensor(pose_rpy):
    pose_rpy_norm = pose_rpy.detach().clone()
  else:
    pose_rpy_norm = copy.deepcopy(pose_rpy)
  pose_rpy_norm[:, 0] = pose_rpy_norm[:, 0] - 0.3
  pose_rpy_norm[:, :3] = pose_rpy_norm[:, :3] * 100
  pose_rpy_norm[:, 3:] = pose_rpy_norm[:, 3:] / 180 / np.pi
  return pose_rpy_norm

def reverse_pose_rpy_v1(pose_rpy_norm):
  if torch.is_tensor(pose_rpy_norm):
    pose_rpy = pose_rpy_norm.detach().clone()
  else:
    pose_rpy = copy.deepcopy(pose_rpy_norm)
  pose_rpy[:, :3] = pose_rpy[:, :3] / 100
  pose_rpy[:, 0] = pose_rpy[:, 0] + 0.3
  pose_rpy[:, 3:] = pose_rpy[:, 3:] * 180 * np.pi
  return pose_rpy

def normalize_d_pose_rpy_v1(pose_rpy_norm):
  if torch.is_tensor(pose_rpy_norm):
    d_pose_rpy_norm = torch.zeros_like(pose_rpy_norm)
  else:
      d_pose_rpy_norm = np.zeros_like(pose_rpy_norm)
  d_pose_rpy_norm[:-1] = pose_rpy_norm[1:] - pose_rpy_norm[:-1]
  d_pose_rpy_norm[:, :3] = d_pose_rpy_norm[:, :3] * 10
  d_pose_rpy_norm[:, 3:] = d_pose_rpy_norm[:, 3:] * 1000
  return d_pose_rpy_norm

def reverse_d_pose_rpy_v1(d_pose_rpy_norm):
  if torch.is_tensor(d_pose_rpy_norm):
    d_pose_rpy = d_pose_rpy_norm.detach().clone()
  else:
    d_pose_rpy = copy.deepcopy(d_pose_rpy_norm)
  d_pose_rpy[:, :3] = d_pose_rpy[:, :3] / 10 / 100
  d_pose_rpy[:, 3:] = d_pose_rpy[:, 3:] / 1000 * 180 * np.pi
  return d_pose_rpy

def normalize_pose_quat_v1(pose_quat):
  if torch.is_tensor(pose_quat):
    pose_quat_norm = pose_quat.detach().clone()
  else:
    pose_quat_norm = copy.deepcopy(pose_quat)
  pose_quat_norm[:, 0] = pose_quat_norm[:, 0] - 0.3
  pose_quat_norm[:, :3] = pose_quat_norm[:, :3] * 100
  return pose_quat_norm

def reverse_pose_quat_v1(pose_quat_norm):
  if torch.is_tensor(pose_quat_norm):
    pose_quat = pose_quat_norm.detach().clone()
  else:
    pose_quat = copy.deepcopy(pose_quat_norm)
  pose_quat[:, :3] = pose_quat[:, :3] / 100
  pose_quat[:, 0] = pose_quat[:, 0] + 0.3
  return pose_quat

def normalize_d_pose_quat_v1(pose_quat_norm):
  if torch.is_tensor(pose_quat_norm):
    d_pose_quat_norm = torch.zeros_like(pose_quat_norm)
  else:
    d_pose_quat_norm = np.zeros_like(pose_quat_norm)
  d_pose_quat_norm[:-1] = pose_quat_norm[1:] - pose_quat_norm[:-1]
  d_pose_quat_norm[:, :3] = d_pose_quat_norm[:, :3] * 10
  d_pose_quat_norm[:, 3:] = d_pose_quat_norm[:, 3:] * 1000
  return d_pose_quat_norm


def reverse_d_pose_quat_v1(d_pose_quat_norm):
  if torch.is_tensor(d_pose_quat_norm):
    d_pose_quat = d_pose_quat_norm.detach().clone()
  else:
    d_pose_quat = copy.deepcopy(d_pose_quat_norm)
  d_pose_quat[:, :3] = d_pose_quat[:, :3] / 10 /100
  d_pose_quat[:, 3:] = d_pose_quat[:, 3:] / 1000
  return d_pose_quat


# v2
def normalize_pose_quat_v2(pose_quat):
  if torch.is_tensor(pose_quat):
    pose_quat_norm = pose_quat.detach().clone()
  else:
    pose_quat_norm = copy.deepcopy(pose_quat)
  pose_quat_norm[:, :3] = pose_quat_norm[:, :3] * 25
  return pose_quat_norm

def reverse_pose_quat_v2(pose_quat_norm):
  if torch.is_tensor(pose_quat_norm):
    pose_quat = pose_quat_norm.detach().clone()
  else:
    pose_quat = copy.deepcopy(pose_quat_norm)
  pose_quat[:, :3] = pose_quat[:, :3] / 25
  return pose_quat

def normalize_d_pose_quat_v2(pose_quat_norm):
  if torch.is_tensor(pose_quat_norm):
    d_pose_quat_norm = torch.zeros_like(pose_quat_norm)
  else:
    d_pose_quat_norm = np.zeros_like(pose_quat_norm)
  d_pose_quat_norm[:-1] = pose_quat_norm[1:] - pose_quat_norm[:-1]
  d_pose_quat_norm[:, :3] = d_pose_quat_norm[:, :3] * 2000
  d_pose_quat_norm[:, 3:] = d_pose_quat_norm[:, 3:] * 800
  return d_pose_quat_norm

def reverse_d_pose_quat_v2(d_pose_quat_norm):
  if torch.is_tensor(d_pose_quat_norm):
    d_pose_quat = d_pose_quat_norm.detach().clone()
  else:
    d_pose_quat = copy.deepcopy(d_pose_quat_norm)
  d_pose_quat[:, :3] = d_pose_quat[:, :3] / 2000
  d_pose_quat[:, 3:] = d_pose_quat[:, 3:] / 800
  return d_pose_quat


def postprocess_pose(name, pose):
  if "d_pose_rpy_norm" in name:
    return reverse_d_pose_rpy_v1(pose)
  elif "pose_rpy_norm" in name:
    return reverse_pose_rpy_v1(pose)
  elif "d_pose_quat_norm" in name:
    return reverse_d_pose_quat_v1(pose)
  elif "pose_quat_norm" in name:
    return reverse_pose_quat_v1(pose)
  elif "d_pose_quat_v2" in name:
    return reverse_d_pose_quat_v2(pose)
  elif "pose_quat_v2" in name:
    return reverse_pose_quat_v2(pose)
  else:
    return pose
  

def preprocess_pose(data):
  if "pose_rpy" in data.keys():
    data["pose_rpy_next"] = shift_next_time(data["pose_rpy"])

    # v1
    data["pose_rpy_norm"] = normalize_pose_rpy_v1(data["pose_rpy"])
    data["pose_rpy_norm_next"] = shift_next_time(data["pose_rpy_norm"])
    data["d_pose_rpy_norm"] = normalize_d_pose_rpy_v1(data["pose_rpy_norm"])
    data["d_pose_rpy_norm_prev"] = shift_prev_time(data["d_pose_rpy_norm"])

  if "pose_quat" in data.keys():
    data["pose_quat_next"] = shift_next_time(data["pose_quat"])

    # v1
    data["pose_quat_norm"] = normalize_pose_quat_v1(data["pose_quat"])
    data["pose_quat_norm_next"] = shift_next_time(data["pose_quat_norm"])
    data["d_pose_quat_norm"] = normalize_d_pose_quat_v1(data["pose_quat_norm"])
    data["d_pose_quat_norm_prev"] = shift_prev_time(data["d_pose_quat_norm"])
    # v2
    data["pose_quat_v2"] = normalize_pose_quat_v2(data["pose_quat"])
    data["pose_quat_v2_next"] = shift_next_time(data["pose_quat_v2"])
    data["d_pose_quat_v2"] = normalize_d_pose_quat_v2(data["pose_quat"])
    data["d_pose_quat_v2_prev"] = shift_prev_time(data["d_pose_quat_v2"])
  
  if "servo_value" in data.keys():
    data["servo_value_next"] = shift_next_time(data["servo_value"])
  return data

def preprocess_pose_seq(data, pose_prev):
  if "pose_quat" in data.keys():
    action_size = pose_prev.size(1)
    data["pose_quat_v2"] = normalize_pose_quat_v2(data["pose_quat"][:,:action_size])

    data["d_pose_quat_v2_prev"] = data["pose_quat"][:,:action_size] - pose_prev
    data["d_pose_quat_v2_prev"][:, :3] = data["d_pose_quat_v2_prev"][:, :3] * 2000
    data["d_pose_quat_v2_prev"][:, 3:] = data["d_pose_quat_v2_prev"][:, 3:] * 800
  return data
