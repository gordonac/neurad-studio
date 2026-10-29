# Copyright 2026 the authors of NeuRAD and contributors.
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

from pathlib import Path

import numpy as np
import pytest

# py123d is an optional dependency; skip these tests when it is not installed.
pytest.importorskip("py123d", reason="py123d is an optional dependency for these tests")

from nerfstudio.data.dataparsers.py123d_utils import (
    DEFAULT_CAPTURE_METADATA_MODALITY,
    WLH_TO_LWH,
    bounding_box_se3_from_list,
    build_camera_name_to_modality_key,
    decode_capture_metadata_payload,
    get_camera_capture_timestamp_us,
    get_pinhole_camera_metadata,
    is_label_prefix_allowed,
    lidar_ipc_to_numpy,
    load_capture_metadata_table,
    pinhole_distortion_to_neurad,
    pose_list_to_matrix,
    read_arrow_table,
)


def test_pinhole_distortion_to_neurad() -> None:
    coeffs = pinhole_distortion_to_neurad([-0.4, 0.22, 0.0007, 0.0004, -0.096])
    assert coeffs.tolist() == pytest.approx([-0.4, 0.22, -0.096, 0.0, 0.0007, 0.0004])


def test_pose_list_to_matrix_is_valid_rigid_transform() -> None:
    matrix = pose_list_to_matrix([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    assert matrix.shape == (4, 4)
    assert matrix[3, 3] == 1.0
    rotation = matrix[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)


def test_is_label_prefix_allowed() -> None:
    allowed = {"vehicle", "person", "two_wheeler"}
    assert is_label_prefix_allowed("vehicle", allowed)
    assert is_label_prefix_allowed("vehicle.emergency", allowed)
    assert is_label_prefix_allowed("person", allowed)
    assert is_label_prefix_allowed("two_wheeler", allowed)
    assert not is_label_prefix_allowed("animal", allowed)
    assert not is_label_prefix_allowed("other", allowed)


def test_bounding_box_se3_from_list_and_wlh_rotation() -> None:
    box = bounding_box_se3_from_list([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5])
    assert box.length == pytest.approx(4.0)
    assert box.width == pytest.approx(2.0)
    assert box.height == pytest.approx(1.5)
    pose = box.center_se3.transformation_matrix @ WLH_TO_LWH
    assert pose.shape == (4, 4)
    assert np.allclose(pose[:3, 3], [1.0, 2.0, 3.0])


def test_decode_capture_metadata_payload_dict_passthrough() -> None:
    payload = {"cameras": {"CAM_FRONT": {"capture_timestamp_us": 123}}}
    assert decode_capture_metadata_payload(payload) is payload


def test_mini_capture_metadata_sidecar(mini_log: Path) -> None:
    capture = load_capture_metadata_table(mini_log, DEFAULT_CAPTURE_METADATA_MODALITY)
    assert capture is not None
    sync = read_arrow_table(mini_log / "sync.arrow")
    ts = get_camera_capture_timestamp_us(
        capture, sync, DEFAULT_CAPTURE_METADATA_MODALITY, sync_row=0, camera_name="CAM_FRONT"
    )
    assert ts is not None
    assert (
        get_camera_capture_timestamp_us(
            capture, sync, DEFAULT_CAPTURE_METADATA_MODALITY, sync_row=0, camera_name="MISSING"
        )
        is None
    )
    assert load_capture_metadata_table(mini_log, "custom.does_not_exist") is None


def test_mini_log_camera_metadata_and_lidar_decode(mini_log: Path) -> None:
    mapping = build_camera_name_to_modality_key(mini_log)
    assert "CAM_FRONT" in mapping

    metadata = get_pinhole_camera_metadata(mini_log / f"{mapping['CAM_FRONT']}.arrow")
    assert metadata.camera_name == "CAM_FRONT"
    assert metadata.intrinsics.fx > 0

    lidar_table = read_arrow_table(mini_log / "lidar.lidar_merged.arrow")
    data = lidar_table.column(2)[0].as_py()
    points = lidar_ipc_to_numpy(data, sweep_duration_s=0.05)
    assert points.shape[1] == 6
    assert points.shape[0] > 0


def test_lidar_point_times_are_built_per_sensor(make_lidar_ipc_bytes) -> None:
    """Every sensor in a merged cloud spins across the whole sweep, so every sensor gets the whole ramp.

    A merged modality stores one sensor's points after another's. Ramping across the concatenation
    gives the first sensor the start of the sweep and the last sensor only its final milliseconds --
    here, with 40 + 10 + 25 points, sensor 2 would be stamped within the last 17 ms of a 50 ms sweep.
    """
    points_per_sensor = (40, 10, 25)
    sweep_duration_s = 0.05
    points = lidar_ipc_to_numpy(make_lidar_ipc_bytes(points_per_sensor), sweep_duration_s=sweep_duration_s)

    assert points.shape[0] == sum(points_per_sensor)
    times = points[:, 4]
    start = 0
    for count in points_per_sensor:
        block = times[start : start + count]
        assert block.min() == pytest.approx(-sweep_duration_s)
        assert block.max() == pytest.approx(0.0)
        assert np.all(np.diff(block) > 0), "times must still advance with firing order within a sensor"
        start += count


def test_lidar_point_times_unchanged_for_a_single_sensor(make_lidar_ipc_bytes) -> None:
    """One sensor, or no ``ids`` column at all, keeps the plain uniform ramp."""
    points = lidar_ipc_to_numpy(make_lidar_ipc_bytes(num_points=8), sweep_duration_s=0.05)
    np.testing.assert_allclose(points[:, 4], np.linspace(-0.05, 0.0, 8), atol=1e-7)


def test_py123d_dataparser_smoke(mini_log: Path) -> None:
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = Py123dDataParserConfig(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT", "CAM_FRONT_LEFT"),
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )
    parser = Py123dDataParser(config)
    outputs = parser.get_dataparser_outputs("train")

    assert len(outputs.image_filenames) == 3 * len(config.cameras)
    assert outputs.cameras.shape[0] == len(outputs.image_filenames)
    assert outputs.metadata["lidars"].shape[0] == 3
    assert len(outputs.metadata["point_clouds"]) == 3
    assert outputs.metadata["point_clouds"][0].shape[1] == 6
    assert outputs.cameras.distortion_params is not None
    assert torch.all(outputs.cameras.distortion_params[:, 3] == 0)


def test_py123d_rejects_multiple_lidars(mini_log: Path) -> None:
    pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = Py123dDataParserConfig(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT",),
        lidars=("LIDAR_TOP", "LIDAR_REAR"),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )
    with pytest.raises(ValueError, match="exactly one lidar name"):
        Py123dDataParser(config).get_dataparser_outputs("train")


def test_py123d_actor_trajectories(mini_log: Path) -> None:
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = Py123dDataParserConfig(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT",),
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        load_cuboids=True,
        train_split_fraction=1.0,
    )
    parser = Py123dDataParser(config)
    outputs = parser.get_dataparser_outputs("train")

    trajectories = outputs.metadata["trajectories"]
    assert len(trajectories) > 0
    traj = trajectories[0]
    assert traj["poses"].ndim == 3 and traj["poses"].shape[-2:] == (4, 4)
    assert traj["timestamps"].ndim == 1
    assert traj["dims"].shape == (3,)
    assert traj["label"] == "vehicle"
    assert traj["stationary"] is False
    assert traj["deformable"] is False
    assert torch.is_floating_point(traj["poses"])


def test_py123d_per_camera_capture_timestamps(mini_log: Path) -> None:
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    cameras = ("CAM_FRONT", "CAM_FRONT_LEFT")
    common = dict(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=cameras,
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )

    with_capture = Py123dDataParser(
        Py123dDataParserConfig(**common, use_capture_timestamps=True)
    ).get_dataparser_outputs("train")
    without_capture = Py123dDataParser(
        Py123dDataParserConfig(**common, use_capture_timestamps=False)
    ).get_dataparser_outputs("train")

    n = 3
    capture_f = with_capture.cameras.times[:n].squeeze(-1)
    capture_l = with_capture.cameras.times[n : 2 * n].squeeze(-1)
    anchor_f = without_capture.cameras.times[:n].squeeze(-1)
    anchor_l = without_capture.cameras.times[n : 2 * n].squeeze(-1)

    assert not torch.allclose(capture_f, capture_l)
    assert torch.allclose(anchor_f, anchor_l)
    assert not torch.allclose(capture_f, anchor_f)
    delta_ms = (capture_f - anchor_f).abs() * 1e3
    assert float(delta_ms.mean()) == pytest.approx(10.0, abs=0.1)


def test_py123d_handles_null_sync_references(make_py123d_log) -> None:
    """A sample with no frame from one sensor must be skipped, not crash the parse.

    ``sync.arrow`` entries are null whenever a synchronized sample has no data for that
    modality. One camera effectively defines the sync grid and never produces a null;
    several cameras at staggered capture rates produce them routinely.
    """
    pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    log_dir = make_py123d_log(null_sync_samples={"CAM_FRONT_LEFT": [2], "lidar": [0]})
    config = Py123dDataParserConfig(
        data=log_dir.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT", "CAM_FRONT_LEFT"),
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=1.0,
    )
    parser = Py123dDataParser(config)  # the images live as long as the parser does
    outputs = parser.get_dataparser_outputs("train")

    # 3 CAM_FRONT frames + 2 CAM_FRONT_LEFT frames, and 2 of 3 lidar sweeps.
    assert len(outputs.image_filenames) == 5
    assert outputs.cameras.shape[0] == 5
    assert all(path.exists() for path in outputs.image_filenames)
    assert outputs.metadata["lidars"].shape[0] == 2
    assert len(outputs.metadata["point_clouds"]) == 2


def test_py123d_image_paths_survive_a_second_split(mini_log: Path) -> None:
    """Paths handed out for one split must still be readable after the next split parses."""
    pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    config = Py123dDataParserConfig(
        data=mini_log.parents[2],
        log_id="mini_log_001",
        split="train",
        cameras=("CAM_FRONT",),
        lidars=("LIDAR_TOP",),
        lidar_type=LidarType.VELODYNE_VLP32C,
        train_split_fraction=0.5,
    )
    parser = Py123dDataParser(config)

    train = parser.get_dataparser_outputs("train")
    train_paths = list(train.image_filenames)
    assert train_paths and all(path.exists() for path in train_paths)

    val = parser.get_dataparser_outputs("val")

    assert all(path.exists() for path in train_paths), "train images deleted by the eval parse"
    assert all(path.exists() for path in val.image_filenames)


def test_py123d_georeferenced_poses_keep_their_precision(make_py123d_log) -> None:
    """The same scene must come out the same whether or not it is georeferenced.

    Real py123d logs are georeferenced, with ego translations of order 1e6 m, where
    float32 spacing is 0.25 m. Casting poses to float32 in that frame quantizes every
    camera, lidar and actor pose independently; recentering the scene afterwards cannot
    recover it. The public example datasets all ship near-zero coordinates, so this is
    invisible without a log like the one built here.
    """
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    def parse(origin):
        log_dir = make_py123d_log(num_samples=6, global_origin=origin)
        config = Py123dDataParserConfig(
            data=log_dir.parents[2],
            log_id="mini_log_001",
            split="train",
            cameras=("CAM_FRONT", "CAM_FRONT_LEFT"),
            lidars=("LIDAR_TOP",),
            lidar_type=LidarType.VELODYNE_VLP32C,
            load_cuboids=True,
            train_split_fraction=1.0,
        )
        return Py123dDataParser(config).get_dataparser_outputs("train")

    at_origin = parse((0.0, 0.0, 0.0))
    georeferenced = parse((3.8e6, 1.2e6, 40.0))

    # Both logs describe the same scene, and the parser recenters on the mean pose, so
    # every recentered pose must agree. Without the float64 pre-shift the georeferenced
    # positions are off by ~0.06 m on average.
    cam_error = (georeferenced.cameras.camera_to_worlds[:, :3, 3] - at_origin.cameras.camera_to_worlds[:, :3, 3]).abs()
    lidar_error = (
        georeferenced.metadata["lidars"].lidar_to_worlds[:, :3, 3]
        - at_origin.metadata["lidars"].lidar_to_worlds[:, :3, 3]
    ).abs()
    actor_error = (
        georeferenced.metadata["trajectories"][0]["poses"][:, :3, 3]
        - at_origin.metadata["trajectories"][0]["poses"][:, :3, 3]
    ).abs()

    assert float(cam_error.max()) < 1e-3
    assert float(lidar_error.max()) < 1e-3
    assert float(actor_error.max()) < 1e-3

    # dataparser_transform still maps the exporter's global frame into the world frame.
    transform = georeferenced.dataparser_transform.double()
    global_pose = torch.tensor([3.8e6, 1.2e6, 40.0, 1.0], dtype=torch.float64)
    world_pose = transform @ global_pose
    assert float(world_pose.abs().max()) < 10.0, world_pose


def test_py123d_lidar_sensor_offset_is_a_frame_change_only(mini_log: Path) -> None:
    """`lidar_sensor_offset` must move points into the sensor frame without moving the world."""
    torch = pytest.importorskip("torch")

    from nerfstudio.cameras.lidars import LidarType
    from nerfstudio.data.dataparsers.py123d_dataparser import Py123dDataParser, Py123dDataParserConfig

    offset = (0.0, 0.0, 1.5)

    def parse(lidar_sensor_offset):
        config = Py123dDataParserConfig(
            data=mini_log.parents[2],
            log_id="mini_log_001",
            split="train",
            cameras=("CAM_FRONT",),
            lidars=("LIDAR_TOP",),
            lidar_type=LidarType.VELODYNE_VLP32C,
            lidar_sensor_offset=lidar_sensor_offset,
            train_split_fraction=1.0,
        )
        return Py123dDataParser(config).get_dataparser_outputs("train")

    stored_frame = parse((0.0, 0.0, 0.0))
    sensor_frame = parse(offset)

    offset_tensor = torch.tensor(offset)
    for stored_pc, sensor_pc in zip(stored_frame.metadata["point_clouds"], sensor_frame.metadata["point_clouds"]):
        # The points themselves are shifted into the sensor frame...
        assert torch.allclose(sensor_pc[:, :3], stored_pc[:, :3] - offset_tensor, atol=1e-6)

    def world_points(outputs):
        """Points in world space, relative to the first camera of that parse.

        Each parse recenters the scene on its own mean pose, so an absolute comparison
        would only measure that recentering. The cameras are untouched by the lidar
        offset, so camera 0 is the fixed landmark; the lidar origin is not, since moving
        it to the physical sensor is the whole point of the config field.
        """
        lidar_to_worlds = outputs.metadata["lidars"].lidar_to_worlds
        anchor = outputs.cameras.camera_to_worlds[0, :3, 3]
        out = []
        for idx, pc in enumerate(outputs.metadata["point_clouds"]):
            rotation = lidar_to_worlds[idx, :3, :3]
            translation = lidar_to_worlds[idx, :3, 3]
            out.append(pc[:, :3] @ rotation.T + translation - anchor)
        return torch.cat(out)

    assert torch.allclose(world_points(sensor_frame), world_points(stored_frame), atol=1e-5)

    # And the default leaves both the points and the poses untouched.
    default = parse((0.0, 0.0, 0.0))
    assert torch.allclose(default.metadata["lidars"].lidar_to_worlds, stored_frame.metadata["lidars"].lidar_to_worlds)
