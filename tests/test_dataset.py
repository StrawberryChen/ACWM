import torch

from dataset.trajectory import Trajectory, TrajectoryDataset


def test_trajectory_window_alignment():
    frames = torch.arange(6 * 3 * 8 * 8).reshape(6, 3, 8, 8).float()
    actions = torch.arange(10).reshape(5, 2).float()
    dataset = TrajectoryDataset([Trajectory(frames, actions)], history_length=3)
    sample = dataset[0]
    assert len(dataset) == 3
    assert torch.equal(sample["history_frames"], frames[:3])
    assert torch.equal(sample["history_actions"], actions[:2])
    assert torch.equal(sample["current_action"], actions[2])
    assert torch.equal(sample["next_history_frames"], frames[1:4])
    assert torch.equal(sample["next_history_actions"], actions[1:3])


def test_rollout_window():
    frames = torch.rand(7, 3, 8, 8)
    actions = torch.rand(6, 2)
    dataset = TrajectoryDataset([Trajectory(frames, actions)], history_length=3, rollout_length=2)
    sample = dataset[0]
    assert len(dataset) == 3
    assert torch.equal(sample["rollout_actions"], actions[2:4])
    assert torch.equal(sample["goal_frame"], frames[4])
