#!/usr/bin/env python3
"""
使用 reBot Arm 102 主臂录制动作，并在 reBot B601-RS 从臂上回放。
Record motions with a reBot Arm 102 leader and replay them on a reBot B601-RS follower.

交互方式保持与 DM 机械臂录制/播放示例一致：
The controls follow the original DM record/play demo:
  q/w/e/r/t：开始/停止录制第 1～5 个动作槽位 / Start or stop recording slots 1–5
  1/2/3/4/5：播放第 1～5 个动作槽位 / Play slots 1–5
  s：停止录制或播放，并安全返回实时跟随 / Stop and safely return to live follow
  c：清除当前选中的动作槽位 / Clear the selected slot
  a：清除全部动作槽位 / Clear all slots
  f：实时主从跟随模式 / Live-follow mode
  Esc：安全回零后退出 / Return safely to zero and exit

录制数据仅保存在当前程序内存中，与原始示例保持一致。
Recordings are stored only in memory for the current process, matching the original demo.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import logging
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pynput import keyboard

from lerobot_robot_seeed_b601 import (
    SeeedB601RSFollower,
    SeeedB601RSFollowerConfig,
)
from lerobot_teleoperator_rebot_arm_102 import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
)

logger = logging.getLogger("rebot_b601_rs_record_dance")

# =============================================================================
# 用户配置区 / USER CONFIGURATION
# =============================================================================
# 平时只需要修改这一部分。命令行参数仍可临时覆盖这些默认值。
# Normally, edit only this section. Command-line arguments can still override it.

# ----- 1. 硬件连接 / Hardware connection -----
CFG_LEADER_PORT = "/dev/ttyUSB0"
CFG_LEADER_ID = "rebot_arm_102_leader"
CFG_LEADER_BAUDRATE = 1_000_000

CFG_FOLLOWER_PORT = "can0"
CFG_FOLLOWER_ID = "follower1"
CFG_CAN_ADAPTER = "socketcan"

# 断开程序时是否关闭电机扭矩。
# Whether to disable motor torque when the follower disconnects.
CFG_DISABLE_TORQUE_ON_DISCONNECT = True

# ----- 2. 主控制与终端显示 / Main control and terminal display -----
CFG_CONTROL_HZ = 30.0

# 0 或负数：关闭 LeRobot 的单周期相对目标限幅。
# Positive value: maximum target change in degrees per control cycle.
CFG_MAX_RELATIVE_TARGET_DEG = 0.0

CFG_PLAY_LOOP = True
CFG_PRINT_ACTIONS = False
CFG_LOG_LEVEL = "INFO"

# 实时角度与温度显示。
# Live angle and temperature telemetry.
CFG_TELEMETRY_ENABLED = True
CFG_TELEMETRY_HZ = 2.0

# ----- 3. 温度监控 / Temperature monitoring -----
# 达到警告温度时只显示标记，不停机。
# Warning only; no shutdown.
CFG_TEMP_WARNING_C = 70.0

# 分级显示阈值。
# Staged visual warning thresholds.
CFG_TEMP_ALERT_LEVEL_2_C = 85.0
CFG_TEMP_ALERT_LEVEL_3_C = 110.0
CFG_TEMP_ALERT_LEVEL_4_C = 125.0

# 任一电机的 MOS 或转子温度达到该值时自动停机。
# Automatic shutdown when either MOS or rotor temperature reaches this value.
CFG_TEMP_CRITICAL_C = 130.0

# ----- 4. 录制与回放 / Recording and playback -----
CFG_NUM_SLOTS = 5
CFG_END_HOLD_TIME_S = 0.15

# 动作末尾回到开头的平滑衔接。
# Smooth loop end-to-start transition.
CFG_LOOP_BLEND_MIN_TIME_S = 0.60
CFG_LOOP_BLEND_ARM_MAX_SPEED_DEG_S = 15.0
CFG_LOOP_BLEND_GRIPPER_MAX_SPEED_DEG_S = 5.0

CFG_RECORD_FILTER_ALPHA = 0.35
CFG_MIN_RECORD_INTERVAL_S = 0.01
CFG_MIN_JOINT_CHANGE_DEG = 0.172

# 播放不同槽位时，移动到新动作第一帧的固定过渡时间。
# Transition time to the first frame when switching playback slots.
CFG_TRANSITION_TIME_S = 0.60

# ----- 5. 播放/录制切换到主臂时的安全同步 / Safe synchronization -----
CFG_LEADER_SYNC_ARM_MAX_SPEED_DEG_S = 15.0
CFG_LEADER_SYNC_GRIPPER_MAX_SPEED_DEG_S = 5.0
CFG_LEADER_SYNC_TOLERANCE_DEG = 0.50

# ----- 6. Esc/Ctrl+C 安全回零 / Safe return-to-zero -----
CFG_RETURN_ZERO_ARM_MAX_SPEED_DEG_S = 15.0
CFG_RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S = 5.0
CFG_RETURN_ZERO_MIN_TIME_S = 3.0
CFG_RETURN_ZERO_SETTLE_TIME_S = 0.30

# 未按 Esc/Ctrl+C、而是主循环自然结束时，是否也回零。
# Whether a normal non-Esc/non-Ctrl+C exit should also return to zero.
CFG_RETURN_ZERO_ON_NORMAL_EXIT = False
# =============================================================================


class _RelativeGoalClampFilter(logging.Filter):
    """Hide only LeRobot's high-frequency relative-goal clamp warning.

    The safety clamp itself remains active. CAN errors, motor faults, temperature
    warnings, and all other log records are left untouched.
    """

    MESSAGE_PREFIX = (
        "Relative goal position magnitude had to be clamped to be safe."
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return not record.getMessage().startswith(self.MESSAGE_PREFIX)
        except Exception:
            return True


def _silence_repeated_relative_goal_clamp_warning() -> None:
    """Install the targeted filter on the root logger and its handlers."""
    root_logger = logging.getLogger()
    clamp_filter = _RelativeGoalClampFilter()
    root_logger.addFilter(clamp_filter)
    for handler in root_logger.handlers:
        handler.addFilter(clamp_filter)


ACTION_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_yaw.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)

MOTOR_SHORT_NAMES = {
    "shoulder_pan": "J1",
    "shoulder_lift": "J2",
    "elbow_flex": "J3",
    "wrist_flex": "J4",
    "wrist_yaw": "J5",
    "wrist_roll": "J6",
    "gripper": "G",
}

RECORD_KEYS = {"q": 0, "w": 1, "e": 2, "r": 3, "t": 4}
PLAY_KEYS = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


class Mode(str, Enum):
    FOLLOW = "follow"
    RECORD = "record"
    SYNC_TO_LEADER = "sync_to_leader"
    TRANSITION = "transition"
    PLAYBACK = "playback"


MODE_CN_LABELS = {
    Mode.FOLLOW: "实时跟随 / FOLLOW",
    Mode.RECORD: "正在录制 / RECORD",
    Mode.SYNC_TO_LEADER: "安全同步 / SYNC",
    Mode.TRANSITION: "平滑过渡 / TRANSITION",
    Mode.PLAYBACK: "正在播放 / PLAYBACK",
}


@dataclass
class MotionFrame:
    t: float
    action: dict[str, float]


class RSDanceRecorder:
    NUM_SLOTS = CFG_NUM_SLOTS

    # Kept close to the original DM demo.
    END_HOLD_TIME_S = CFG_END_HOLD_TIME_S

    # 循环播放末尾回到开头时，不再使用固定的短过渡时间。
    # 根据首尾姿态差自动延长过渡，限制 smoothstep 的峰值速度。
    LOOP_BLEND_MIN_TIME_S = CFG_LOOP_BLEND_MIN_TIME_S
    LOOP_BLEND_ARM_MAX_SPEED_DEG_S = CFG_LOOP_BLEND_ARM_MAX_SPEED_DEG_S
    LOOP_BLEND_GRIPPER_MAX_SPEED_DEG_S = CFG_LOOP_BLEND_GRIPPER_MAX_SPEED_DEG_S

    RECORD_FILTER_ALPHA = CFG_RECORD_FILTER_ALPHA
    MIN_RECORD_INTERVAL_S = CFG_MIN_RECORD_INTERVAL_S
    # Original demo used 0.003 rad; this script records degrees.
    MIN_JOINT_CHANGE_DEG = CFG_MIN_JOINT_CHANGE_DEG
    TRANSITION_TIME_S = CFG_TRANSITION_TIME_S

    # Safe playback -> leader synchronization. The bottom-layer relative-goal
    # clamp is disabled for smooth replay, so mode switching is rate-limited here.
    LEADER_SYNC_ARM_MAX_SPEED_DEG_S = CFG_LEADER_SYNC_ARM_MAX_SPEED_DEG_S
    LEADER_SYNC_GRIPPER_MAX_SPEED_DEG_S = CFG_LEADER_SYNC_GRIPPER_MAX_SPEED_DEG_S
    LEADER_SYNC_TOLERANCE_DEG = CFG_LEADER_SYNC_TOLERANCE_DEG

    # Graceful Ctrl+C return-to-zero settings.  The actual return duration is
    # automatically extended so that no action-space joint exceeds this speed.
    RETURN_ZERO_ARM_MAX_SPEED_DEG_S = CFG_RETURN_ZERO_ARM_MAX_SPEED_DEG_S
    # The configured gripper action is multiplied by 6 before reaching the
    # motor, so keep its action-space return speed lower.
    RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S = CFG_RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S
    RETURN_ZERO_MIN_TIME_S = CFG_RETURN_ZERO_MIN_TIME_S
    RETURN_ZERO_SETTLE_TIME_S = CFG_RETURN_ZERO_SETTLE_TIME_S

    def __init__(
        self,
        leader: RebotArm102Leader,
        follower: SeeedB601RSFollower,
        *,
        control_hz: float = CFG_CONTROL_HZ,
        play_loop: bool = CFG_PLAY_LOOP,
        return_zero_on_exit: bool = CFG_RETURN_ZERO_ON_NORMAL_EXIT,
        print_actions: bool = CFG_PRINT_ACTIONS,
        telemetry_enabled: bool = CFG_TELEMETRY_ENABLED,
        telemetry_hz: float = CFG_TELEMETRY_HZ,
        temp_warning_c: float = CFG_TEMP_WARNING_C,
        temp_critical_c: float = CFG_TEMP_CRITICAL_C,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz 必须大于 0 / control_hz must be greater than zero")

        self.leader = leader
        self.follower = follower
        self.control_hz = float(control_hz)
        self.control_period_s = 1.0 / self.control_hz
        self.play_loop = bool(play_loop)
        self.return_zero_on_exit = bool(return_zero_on_exit)
        self.print_actions = bool(print_actions)

        if telemetry_hz <= 0:
            raise ValueError(
                "telemetry_hz 必须大于 0 / telemetry_hz must be greater than zero"
            )
        if temp_critical_c <= temp_warning_c:
            raise ValueError(
                "危险停机温度必须高于警告温度 / "
                "temp_critical_c must be greater than temp_warning_c"
            )

        self.telemetry_enabled = bool(telemetry_enabled)
        self.telemetry_hz = float(telemetry_hz)
        self.telemetry_period_s = 1.0 / self.telemetry_hz
        self.temp_warning_c = float(temp_warning_c)
        self.temp_critical_c = float(temp_critical_c)
        self.thermal_shutdown_requested = False
        self.thermal_shutdown_motor: str | None = None
        self._last_telemetry_line_length = 0

        self.motion_slots: list[list[MotionFrame]] = [
            [] for _ in range(self.NUM_SLOTS)
        ]

        self.mode = Mode.FOLLOW
        self.selected_slot = 0
        self.running = True

        self.record_slot: int | None = None
        self.record_start_time = 0.0
        self.last_record_time = 0.0
        self.last_record_action: dict[str, float] | None = None
        self.filtered_record_action: dict[str, float] | None = None

        self.play_slot: int | None = None
        self.play_start_time = 0.0
        self.play_frame_times: list[float] = []

        self.transition_start_time = 0.0
        self.transition_from: dict[str, float] | None = None
        self.transition_to: dict[str, float] | None = None

        # Requested record slot while safely synchronizing from playback to
        # the live leader pose. The old slot is not cleared until alignment.
        self.pending_record_slot: int | None = None

        self.last_sent_action: dict[str, float] | None = None
        self.loop_counter = 0

        # First Ctrl+C requests a graceful return to zero.  A second Ctrl+C
        # sets this flag so the user can abort the return immediately.
        self.abort_return_to_zero = False

        self._lock = threading.RLock()
        self._listener: keyboard.Listener | None = None
        self._last_key_time: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Action-space helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_action(action: dict[str, Any]) -> dict[str, float]:
        """Keep only the seven expected position keys and coerce values to float."""
        missing = [key for key in ACTION_KEYS if key not in action]
        if missing:
            raise KeyError(f"动作数据缺少以下关节键 / Action is missing keys: {missing}")
        return {key: float(action[key]) for key in ACTION_KEYS}

    def _observation_to_action_space(
        self, observation: dict[str, Any]
    ) -> dict[str, float]:
        """
        Convert physical follower positions back to the leader/action coordinate space.

        SeeedB601FollowerBase.send_action() applies joint_directions before clipping,
        so starting a transition from a follower observation requires the inverse map.
        This is especially important for the gripper, whose configured scale is 6.
        """
        action: dict[str, float] = {}
        for key in ACTION_KEYS:
            motor_name = key.removesuffix(".pos")
            physical_pos = float(observation.get(key, 0.0))
            direction = float(self.follower.config.joint_directions.get(motor_name, 1.0))
            if abs(direction) < 1e-12:
                raise ValueError(f"joint_directions[{motor_name!r}] 不能为 0 / must not be zero")
            action[key] = physical_pos / direction
        return action

    def _send_action(self, action: dict[str, float]) -> None:
        clean = self._clean_action(action)
        applied_physical = self.follower.send_action(clean)

        # send_action returns physical, direction-applied positions. Convert them
        # back so future transitions still operate in leader/action coordinates.
        self.last_sent_action = self._observation_to_action_space(applied_physical)

    @staticmethod
    def _lerp_action(
        start: dict[str, float], end: dict[str, float], alpha: float
    ) -> dict[str, float]:
        alpha = max(0.0, min(1.0, float(alpha)))
        return {
            key: start[key] + (end[key] - start[key]) * alpha
            for key in ACTION_KEYS
        }

    @staticmethod
    def _smoothstep(alpha: float) -> float:
        alpha = max(0.0, min(1.0, float(alpha)))
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def _low_pass_action(
        self,
        previous: dict[str, float] | None,
        current: dict[str, float],
    ) -> dict[str, float]:
        if previous is None:
            return copy.deepcopy(current)
        alpha = self.RECORD_FILTER_ALPHA
        return {
            key: alpha * current[key] + (1.0 - alpha) * previous[key]
            for key in ACTION_KEYS
        }

    @staticmethod
    def _max_action_delta_deg(
        left: dict[str, float], right: dict[str, float]
    ) -> float:
        return max(abs(left[key] - right[key]) for key in ACTION_KEYS)


    @staticmethod
    def _move_duration_from_speed_limits(
        start: dict[str, float],
        end: dict[str, float],
        *,
        arm_max_speed_deg_s: float,
        gripper_max_speed_deg_s: float,
        minimum_time_s: float,
    ) -> float:
        """按关节距离计算 smoothstep 插值所需时间，限制峰值速度。"""
        if arm_max_speed_deg_s <= 0.0:
            raise ValueError("机械臂关节最大速度必须大于 0")
        if gripper_max_speed_deg_s <= 0.0:
            raise ValueError("夹爪最大速度必须大于 0")

        required_times: list[float] = []
        for key in ACTION_KEYS:
            speed_limit = (
                gripper_max_speed_deg_s
                if key == "gripper.pos"
                else arm_max_speed_deg_s
            )
            # smoothstep 的最大斜率为 1.5，因此乘以 1.5 才能约束峰值速度。
            required_times.append(
                1.5 * abs(end[key] - start[key]) / speed_limit
            )

        return max(float(minimum_time_s), *required_times)

    def _step_toward_leader(
        self,
        current: dict[str, float],
        leader_target: dict[str, float],
    ) -> tuple[dict[str, float], bool]:
        """Move one velocity-limited control step toward the live leader pose."""
        next_action: dict[str, float] = {}
        aligned = True

        for key in ACTION_KEYS:
            delta = leader_target[key] - current[key]
            speed_limit = (
                self.LEADER_SYNC_GRIPPER_MAX_SPEED_DEG_S
                if key == "gripper.pos"
                else self.LEADER_SYNC_ARM_MAX_SPEED_DEG_S
            )
            max_step = speed_limit * self.control_period_s

            if abs(delta) > self.LEADER_SYNC_TOLERANCE_DEG:
                aligned = False

            delta = max(-max_step, min(max_step, delta))
            next_action[key] = current[key] + delta

        return next_action, aligned

    def _begin_leader_sync_locked(
        self,
        *,
        pending_record_slot: int | None,
    ) -> None:
        """Safely leave playback before enabling live leader commands."""
        self.play_slot = None
        self.pending_record_slot = pending_record_slot
        self.mode = Mode.SYNC_TO_LEADER

        if pending_record_slot is None:
            print("\n[同步 / SYNC] 正在缓慢同步到主臂当前位置，完成后恢复实时跟随。 / Smoothly synchronizing to the leader; live follow will resume after alignment.")
        else:
            self.selected_slot = pending_record_slot
            print(
                f"\n[同步 / SYNC] 动作槽位 {pending_record_slot + 1}：正在缓慢同步到主臂位置。 / Slot {pending_record_slot + 1}: smoothly synchronizing to the leader. "
                "同步完成后才开始录制，请暂时保持主臂稳定。 / Recording starts only after alignment; keep the leader steady. "
                "直到出现“开始录制 / RECORDING STARTED”提示。 / Wait for the recording-start message."
            )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _start_recording_locked(self, slot: int, now: float) -> None:
        if self.mode == Mode.RECORD:
            self._stop_recording_locked(now)

        # Playback and playback-transition poses can be far from the leader.
        # Never switch straight to a raw leader command in one control cycle.
        if self.mode in (Mode.PLAYBACK, Mode.TRANSITION, Mode.SYNC_TO_LEADER):
            self._begin_leader_sync_locked(pending_record_slot=slot)
            return

        self._activate_recording_locked(slot, now)

    def _activate_recording_locked(self, slot: int, now: float) -> None:
        """Clear the slot and begin recording only after safe alignment."""
        self.selected_slot = slot
        self.record_slot = slot
        self.pending_record_slot = None
        self.motion_slots[slot] = []
        self.record_start_time = now
        self.last_record_time = -float("inf")
        self.last_record_action = None
        self.filtered_record_action = None
        self.mode = Mode.RECORD
        print(
            f"\n[录制 / RECORD] 动作槽位 {slot + 1}：开始录制。 / Slot {slot + 1}: recording started. "
            "（再次按相同录制键或按 s 停止。） / Press the same record key or s to stop."
        )

    def _record_action_locked(self, action: dict[str, float], now: float) -> None:
        if self.record_slot is None:
            return

        filtered = self._low_pass_action(self.filtered_record_action, action)
        self.filtered_record_action = filtered

        elapsed = now - self.record_start_time
        interval_ok = elapsed - self.last_record_time >= self.MIN_RECORD_INTERVAL_S
        changed_enough = (
            self.last_record_action is None
            or self._max_action_delta_deg(filtered, self.last_record_action)
            >= self.MIN_JOINT_CHANGE_DEG
        )

        if interval_ok and changed_enough:
            self.motion_slots[self.record_slot].append(
                MotionFrame(t=elapsed, action=copy.deepcopy(filtered))
            )
            self.last_record_time = elapsed
            self.last_record_action = copy.deepcopy(filtered)

    def _stop_recording_locked(self, now: float) -> None:
        if self.mode != Mode.RECORD or self.record_slot is None:
            self.mode = Mode.FOLLOW
            self.record_slot = None
            return

        slot = self.record_slot
        frames = self.motion_slots[slot]

        if frames:
            last_t = max(frames[-1].t, now - self.record_start_time)
            # Keep the final pose briefly, as in the DM demo.
            frames.append(
                MotionFrame(
                    t=last_t + self.END_HOLD_TIME_S,
                    action=copy.deepcopy(frames[-1].action),
                )
            )

            # 循环播放时从末尾平滑回到第一帧。
            # 过渡时间按首尾关节距离自动计算，避免固定 0.30 秒造成猛烈回跳。
            if self.play_loop and len(frames) >= 2:
                loop_blend_time_s = self._move_duration_from_speed_limits(
                    frames[-1].action,
                    frames[0].action,
                    arm_max_speed_deg_s=self.LOOP_BLEND_ARM_MAX_SPEED_DEG_S,
                    gripper_max_speed_deg_s=(
                        self.LOOP_BLEND_GRIPPER_MAX_SPEED_DEG_S
                    ),
                    minimum_time_s=self.LOOP_BLEND_MIN_TIME_S,
                )
                frames.append(
                    MotionFrame(
                        t=frames[-1].t + loop_blend_time_s,
                        action=copy.deepcopy(frames[0].action),
                    )
                )
                print(
                    f"\n[录制 / RECORD] 循环首尾平滑过渡时间：{loop_blend_time_s:.2f} 秒。 / Loop end-to-start blend time: {loop_blend_time_s:.2f} s. "
                    f"（机械臂峰值 ≤ {self.LOOP_BLEND_ARM_MAX_SPEED_DEG_S:.1f} 度/秒， / Arm peak ≤ {self.LOOP_BLEND_ARM_MAX_SPEED_DEG_S:.1f} deg/s; "
                    f"夹爪峰值 ≤ {self.LOOP_BLEND_GRIPPER_MAX_SPEED_DEG_S:.1f} 度/秒。） / gripper peak ≤ {self.LOOP_BLEND_GRIPPER_MAX_SPEED_DEG_S:.1f} deg/s."
                )

        print(
            f"\n[录制 / RECORD] 动作槽位 {slot + 1}：录制已停止。 / Slot {slot + 1}: recording stopped. "
            f"共 {len(frames)} 帧， / {len(frames)} frames, "
            f"时长 {frames[-1].t if frames else 0.0:.2f} 秒。 / Duration: {frames[-1].t if frames else 0.0:.2f} s."
        )
        self.record_slot = None
        self.pending_record_slot = None
        self.mode = Mode.FOLLOW

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def _start_playback_locked(self, slot: int, now: float) -> None:
        frames = self.motion_slots[slot]
        if not frames:
            print(f"\n[播放 / PLAY] 动作槽位 {slot + 1} 为空，请先录制动作。 / Slot {slot + 1} is empty; record a motion first.")
            return

        if self.mode == Mode.RECORD:
            self._stop_recording_locked(now)

        self.selected_slot = slot
        self.play_slot = slot
        self.pending_record_slot = None
        self.play_frame_times = [frame.t for frame in frames]

        if self.last_sent_action is None:
            observation = self.follower.get_observation()
            self.last_sent_action = self._observation_to_action_space(observation)

        self.transition_from = copy.deepcopy(self.last_sent_action)
        self.transition_to = copy.deepcopy(frames[0].action)
        self.transition_start_time = now
        self.mode = Mode.TRANSITION
        print(
            f"\n[播放 / PLAY] 动作槽位 {slot + 1}：正在平滑移动到起始姿态。 / Slot {slot + 1}: smoothly moving to the start pose. "
            f"（预计 {self.TRANSITION_TIME_S:.2f} 秒。） / Estimated time: {self.TRANSITION_TIME_S:.2f} s."
        )

    def _playback_action_locked(self, now: float) -> dict[str, float] | None:
        if self.play_slot is None:
            return None
        frames = self.motion_slots[self.play_slot]
        if not frames:
            self.mode = Mode.FOLLOW
            return None

        duration = frames[-1].t
        if duration <= 1e-9:
            return copy.deepcopy(frames[-1].action)

        elapsed = now - self.play_start_time
        if self.play_loop:
            playback_t = elapsed % duration
        else:
            if elapsed >= duration:
                self.mode = Mode.FOLLOW
                print(f"\n[播放 / PLAY] 动作槽位 {self.play_slot + 1}：播放完成。 / Slot {self.play_slot + 1}: playback finished.")
                return copy.deepcopy(frames[-1].action)
            playback_t = elapsed

        right = bisect.bisect_right(self.play_frame_times, playback_t)
        if right <= 0:
            return copy.deepcopy(frames[0].action)
        if right >= len(frames):
            return copy.deepcopy(frames[-1].action)

        left = right - 1
        frame_a = frames[left]
        frame_b = frames[right]
        segment = frame_b.t - frame_a.t
        alpha = 0.0 if segment <= 1e-9 else (playback_t - frame_a.t) / segment
        alpha = self._smoothstep(alpha)
        return self._lerp_action(frame_a.action, frame_b.action, alpha)

    # ------------------------------------------------------------------
    # Keyboard controls
    # ------------------------------------------------------------------
    def _debounced(self, name: str, now: float, debounce_s: float = 0.20) -> bool:
        previous = self._last_key_time.get(name, -float("inf"))
        if now - previous < debounce_s:
            return False
        self._last_key_time[name] = now
        return True

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        now = time.monotonic()

        if key == keyboard.Key.esc:
            print("\n[退出 / EXIT] 检测到 Esc，正在停止动作并缓慢回到零点。 / Esc pressed; stopping motion and returning slowly to zero.")
            # Esc 与第一次 Ctrl+C 使用完全相同的安全退出流程：
            # 先结束主循环，再在 close() 中保持电机使能并缓慢回零，
            # 到达零点后才断开电机。
            self.return_zero_on_exit = True
            self.abort_return_to_zero = False
            self.running = False
            return False

        try:
            char = key.char.lower() if key.char else ""
        except AttributeError:
            return None

        if not char or not self._debounced(char, now):
            return None

        with self._lock:
            if char in RECORD_KEYS:
                slot = RECORD_KEYS[char]
                if self.mode == Mode.RECORD and self.record_slot == slot:
                    self._stop_recording_locked(now)
                elif (
                    self.mode == Mode.SYNC_TO_LEADER
                    and self.pending_record_slot == slot
                ):
                    self.pending_record_slot = None
                    print(
                        f"\n[同步 / SYNC] 动作槽位 {slot + 1}：已取消待录制任务。 / Slot {slot + 1}: pending recording cancelled. "
                        "继续安全同步到实时跟随姿态。 / Continuing safe synchronization to live follow."
                    )
                else:
                    self._start_recording_locked(slot, now)

            elif char in PLAY_KEYS:
                self._start_playback_locked(PLAY_KEYS[char], now)

            elif char == "s":
                if self.mode == Mode.RECORD:
                    self._stop_recording_locked(now)
                elif self.mode in (Mode.PLAYBACK, Mode.TRANSITION):
                    self._begin_leader_sync_locked(pending_record_slot=None)
                elif self.mode == Mode.SYNC_TO_LEADER:
                    self.pending_record_slot = None
                    print(
                        "\n[同步 / SYNC] 已取消待录制任务。 / Pending recording cancelled. "
                        "继续安全同步到实时跟随姿态。 / Continuing safe synchronization to live follow."
                    )
                else:
                    self.mode = Mode.FOLLOW
                    self.play_slot = None
                    print("\n[停止 / STOP] 已停止当前任务，实时跟随已启用。 / Current task stopped; live follow is active.")

            elif char == "f":
                if self.mode == Mode.RECORD:
                    self._stop_recording_locked(now)
                elif self.mode in (Mode.PLAYBACK, Mode.TRANSITION):
                    self._begin_leader_sync_locked(pending_record_slot=None)
                elif self.mode == Mode.SYNC_TO_LEADER:
                    self.pending_record_slot = None
                    print("\n[同步 / SYNC] 正在继续安全同步到实时跟随姿态。 / Continuing safe synchronization to live follow.")
                else:
                    self.mode = Mode.FOLLOW
                    self.play_slot = None
                    print("\n[跟随 / FOLLOW] 已进入实时主从跟随模式。 / Live-follow mode is active.")

            elif char == "c":
                if self.mode == Mode.RECORD and self.record_slot == self.selected_slot:
                    self._stop_recording_locked(now)
                self.motion_slots[self.selected_slot] = []
                print(f"\n[清除 / CLEAR] 已清除动作槽位 {self.selected_slot + 1}。 / Slot {self.selected_slot + 1} cleared.")

            elif char == "a":
                if self.mode == Mode.RECORD:
                    self._stop_recording_locked(now)
                self.motion_slots = [[] for _ in range(self.NUM_SLOTS)]
                self.mode = Mode.FOLLOW
                self.play_slot = None
                print("\n[清除 / CLEAR] 已清除全部动作槽位。 / All slots cleared.")

        return None

    # ------------------------------------------------------------------
    # Main loop / shutdown
    # ------------------------------------------------------------------
    @staticmethod
    def print_help() -> None:
        print(
            "\n"
            "========== reBot B601-RS 动作录制与跳舞回放 / Motion Recorder & Dance Playback ==========\n"
            " q w e r t：先安全同步到主臂，再录制第 1～5 个动作槽位 / Safely sync, then record slots 1–5\n"
            " 1 2 3 4 5：播放第 1～5 个动作槽位 / Play slots 1–5\n"
            " s         ：停止录制/播放，并安全返回实时跟随 / Stop and safely return to live follow\n"
            " f         ：实时主从跟随 / Live follow\n"
            " c         ：清除当前选中的动作槽位 / Clear selected slot\n"
            " a         ：清除全部动作槽位 / Clear all slots\n"
            " Esc       ：停止当前动作并缓慢安全回到零点后退出 / Stop, return safely to zero, and exit\n"
            " Ctrl+C    ：停止当前动作并缓慢安全回到零点 / Stop and return safely to zero\n"
            "========================================================\n"
        )

    def _safe_return_to_zero(self) -> None:
        """Slowly return every action-space joint to zero before disconnecting.

        The duration is derived from the largest joint displacement, so a pose
        far from zero cannot be forced back in a fixed, overly short time.
        """
        # Prefer a fresh physical observation.  Fall back to the last command
        # only when feedback cannot be read during shutdown.
        try:
            observation = self.follower.get_observation()
            start = self._observation_to_action_space(observation)
        except Exception:
            logger.exception(
                "回零前无法读取机械臂当前位置； / Could not read the current pose before return-to-zero; "
                "将使用最后一次发送的目标姿态作为回零起点。 / Using the last commanded pose as the return-to-zero start."
            )
            if self.last_sent_action is None:
                print("[退出 / EXIT] 无法获取当前位置，已跳过自动回零。请手动扶住机械臂并断电。 / Current pose unavailable; automatic return skipped. Hold the arm and cut motor power manually.")
                return
            start = copy.deepcopy(self.last_sent_action)

        target = {key: 0.0 for key in ACTION_KEYS}
        # smoothstep has a peak slope of 1.5, so multiply the nominal
        # delta/speed duration by 1.5 to enforce the requested *peak* speed.
        required_times: list[float] = []
        for key in ACTION_KEYS:
            speed_limit = (
                self.RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S
                if key == "gripper.pos"
                else self.RETURN_ZERO_ARM_MAX_SPEED_DEG_S
            )
            required_times.append(
                1.5 * abs(start[key] - target[key]) / speed_limit
            )

        duration_s = max(self.RETURN_ZERO_MIN_TIME_S, *required_times)
        steps = max(2, int(round(duration_s * self.control_hz)))

        print(
            "[回零 / RETURN] 正在缓慢回到零点。 / Slowly returning to zero. "
            f"预计需要 {duration_s:.1f} 秒。 / Estimated time: {duration_s:.1f} s. "
            f"（机械臂关节峰值速度 ≤ {self.RETURN_ZERO_ARM_MAX_SPEED_DEG_S:.1f} 度/秒， / Arm peak speed ≤ {self.RETURN_ZERO_ARM_MAX_SPEED_DEG_S:.1f} deg/s; "
            f"夹爪峰值速度 ≤ {self.RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S:.1f} 度/秒。） / gripper peak speed ≤ {self.RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S:.1f} deg/s."
        )
        print("[回零 / RETURN] 回零过程中再次按 Ctrl+C，可立即中止并断开电机。 / Press Ctrl+C again to abort immediately and disconnect the motors.")

        next_tick = time.monotonic()
        for i in range(1, steps + 1):
            if self.abort_return_to_zero:
                print("[回零 / RETURN] 检测到第二次 Ctrl+C，已中止回零。 / Second Ctrl+C detected; return-to-zero aborted.")
                return

            alpha = self._smoothstep(i / steps)
            self._send_action(self._lerp_action(start, target, alpha))

            next_tick += self.control_period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        # Send the exact zero target once more and briefly hold it before the
        # follower disconnect disables motor torque.
        self._send_action(target)
        time.sleep(self.RETURN_ZERO_SETTLE_TIME_S)
        print("[回零 / RETURN] 已到达零点，正在断开电机。 / Zero position reached; disconnecting motors.")

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Convert a feedback field to a finite float when possible."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _temperature_marker(self, temperature_c: float | None) -> str:
        """Compact staged warning marker without repeatedly printing log lines."""
        if temperature_c is None:
            return ""
        if temperature_c >= self.temp_critical_c:
            return "XXXXX"
        if temperature_c >= CFG_TEMP_ALERT_LEVEL_4_C:
            return "!!!!"
        if temperature_c >= CFG_TEMP_ALERT_LEVEL_3_C:
            return "!!!"
        if temperature_c >= CFG_TEMP_ALERT_LEVEL_2_C:
            return "!!"
        if temperature_c >= self.temp_warning_c:
            return "!"
        return ""

    def _read_and_display_motor_telemetry(self) -> bool:
        """Read RobStride angle/MOS/rotor temperature and refresh one terminal line.

        Returns False when the configured critical temperature is reached.
        Feedback follows the MotorBridge pattern:
            motor.request_feedback()
            state = motor.get_state()
        """
        # Request all motors first to reduce multi-motor feedback jitter.
        for motor_name in MOTOR_NAMES:
            motor = self.follower.motors.get(motor_name)
            if motor is not None:
                try:
                    motor.request_feedback()
                except Exception:
                    logger.debug(
                        "请求电机反馈失败 / Failed to request feedback: %s",
                        motor_name,
                        exc_info=True,
                    )

        # Newer MotorBridge versions normally poll feedback in the background.
        # Keep one compatibility poll when the bus exposes this method.
        poll_once = getattr(self.follower.bus, "poll_feedback_once", None)
        if callable(poll_once):
            try:
                poll_once()
            except Exception:
                logger.debug(
                    "兼容反馈轮询失败 / Compatibility feedback poll failed",
                    exc_info=True,
                )

        parts: list[str] = []
        critical_events: list[tuple[str, float, float]] = []

        for motor_name in MOTOR_NAMES:
            short_name = MOTOR_SHORT_NAMES[motor_name]
            motor = self.follower.motors.get(motor_name)

            if motor is None:
                parts.append(f"{short_name}:无电机/no motor")
                continue

            try:
                state = motor.get_state()
            except Exception:
                logger.debug(
                    "读取电机状态失败 / Failed to read motor state: %s",
                    motor_name,
                    exc_info=True,
                )
                state = None

            if state is None:
                parts.append(f"{short_name}:无反馈/no data")
                continue

            pos_rad = self._safe_float(getattr(state, "pos", None))
            mos_c = self._safe_float(getattr(state, "t_mos", None))
            rotor_c = self._safe_float(getattr(state, "t_rotor", None))

            pos_deg = math.degrees(pos_rad) if pos_rad is not None else None
            hottest = max(
                (value for value in (mos_c, rotor_c) if value is not None),
                default=None,
            )
            marker = self._temperature_marker(hottest)

            pos_text = "--.-°" if pos_deg is None else f"{pos_deg:+.1f}°"
            mos_text = "--.-" if mos_c is None else f"{mos_c:.1f}"
            rotor_text = "--.-" if rotor_c is None else f"{rotor_c:.1f}"

            parts.append(
                f"{marker}{short_name}:{pos_text} {mos_text}/{rotor_text}C"
            )

            if hottest is not None and hottest >= self.temp_critical_c:
                critical_events.append(
                    (
                        motor_name,
                        mos_c if mos_c is not None else float("nan"),
                        rotor_c if rotor_c is not None else float("nan"),
                    )
                )

        mode_label = MODE_CN_LABELS.get(self.mode, self.mode.value)
        line = (
            f"[状态 / TELEMETRY][{mode_label}] "
            f"角度；温度=MOS/转子 / angle; temp=MOS/rotor | "
            + " | ".join(parts)
        )

        # Overwrite the previous telemetry row instead of continuously adding lines.
        padding = " " * max(0, self._last_telemetry_line_length - len(line))
        print(f"\r{line}{padding}", end="", flush=True)
        self._last_telemetry_line_length = len(line)

        if critical_events:
            print()
            for motor_name, mos_c, rotor_c in critical_events:
                print(
                    "[高温停机 / THERMAL SHUTDOWN] "
                    f"{motor_name} 达到危险温度：MOS={mos_c:.1f}°C，"
                    f"转子={rotor_c:.1f}°C；停机阈值={self.temp_critical_c:.1f}°C。 / "
                    f"{motor_name} reached the critical temperature: "
                    f"MOS={mos_c:.1f}°C, rotor={rotor_c:.1f}°C; "
                    f"shutdown threshold={self.temp_critical_c:.1f}°C."
                )

            print(
                "[高温停机 / THERMAL SHUTDOWN] "
                "正在停止动作并断开电机。高温时不会自动回零，"
                "请立即扶住机械臂。 / "
                "Stopping motion and disconnecting motors. "
                "The arm will not return to zero while overheated; support it immediately."
            )
            self.thermal_shutdown_requested = True
            self.thermal_shutdown_motor = critical_events[0][0]
            # Continuing to move back to zero at 135°C could add more heat.
            self.return_zero_on_exit = False
            self.running = False
            return False

        return True

    def run(self) -> None:
        self.print_help()

        # Initialize last_sent_action from the real follower pose. This prevents
        # the first playback transition from assuming the arm starts at zero.
        observation = self.follower.get_observation()
        self.last_sent_action = self._observation_to_action_space(observation)

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

        next_tick = time.monotonic()
        last_print_time = 0.0
        last_telemetry_time = -float("inf")

        while self.running:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.monotonic()

            # 键盘监听运行在独立线程中。Esc/Ctrl+C 可能在上面的睡眠期间触发，
            # 因此发送下一条动作命令前再次检查，避免退出后多执行一帧。
            if not self.running:
                break

            if now - next_tick > 5 * self.control_period_s:
                # If communication stalls, do not try to execute a large backlog.
                next_tick = now

            with self._lock:
                # Temperature/angle feedback runs at a low rate so it does not
                # compete with the 30 Hz control loop.
                if (
                    self.telemetry_enabled
                    and now - last_telemetry_time >= self.telemetry_period_s
                ):
                    last_telemetry_time = now
                    if not self._read_and_display_motor_telemetry():
                        break

                mode = self.mode

                if mode in (Mode.FOLLOW, Mode.RECORD):
                    leader_action = self._clean_action(self.leader.get_action())
                    self._send_action(leader_action)
                    if mode == Mode.RECORD:
                        self._record_action_locked(leader_action, now)

                elif mode == Mode.SYNC_TO_LEADER:
                    leader_action = self._clean_action(self.leader.get_action())

                    if self.last_sent_action is None:
                        observation = self.follower.get_observation()
                        self.last_sent_action = self._observation_to_action_space(
                            observation
                        )

                    sync_action, aligned = self._step_toward_leader(
                        self.last_sent_action,
                        leader_action,
                    )
                    self._send_action(sync_action)

                    if aligned:
                        pending_slot = self.pending_record_slot
                        if pending_slot is None:
                            self.mode = Mode.FOLLOW
                            print(
                                "\n[同步 / SYNC] 已与主臂位置对齐； / Leader alignment complete; "
                                "已恢复实时跟随。 / Live follow resumed."
                            )
                        else:
                            self._activate_recording_locked(pending_slot, now)
                            # The synchronization path is deliberately excluded
                            # from the recording. Save the aligned pose first.
                            self._record_action_locked(leader_action, now)

                elif mode == Mode.TRANSITION:
                    if self.transition_from is None or self.transition_to is None:
                        self.mode = Mode.FOLLOW
                    else:
                        raw_alpha = (
                            now - self.transition_start_time
                        ) / self.TRANSITION_TIME_S
                        alpha = self._smoothstep(raw_alpha)
                        action = self._lerp_action(
                            self.transition_from,
                            self.transition_to,
                            alpha,
                        )
                        self._send_action(action)
                        if raw_alpha >= 1.0:
                            self.play_start_time = now
                            self.mode = Mode.PLAYBACK
                            print(
                                f"\n[播放 / PLAY] 动作槽位 {self.play_slot + 1 if self.play_slot is not None else '?'}：正在播放。 / Playing slot {self.play_slot + 1 if self.play_slot is not None else '?'}."
                            )

                elif mode == Mode.PLAYBACK:
                    action = self._playback_action_locked(now)
                    if action is not None:
                        self._send_action(action)

                if (
                    self.print_actions
                    and not self.telemetry_enabled
                    and now - last_print_time >= 0.5
                ):
                    last_print_time = now
                    values = "  ".join(
                        f"{key.removesuffix('.pos')}={self.last_sent_action[key]:7.2f}"
                        for key in ACTION_KEYS
                    )
                    mode_label = MODE_CN_LABELS.get(self.mode, self.mode.value)
                    print(f"\r[{mode_label:^8s}] {values}", end="", flush=True)

            self.loop_counter += 1
            next_tick += self.control_period_s

    def close(self, *, normal_exit: bool) -> None:
        self.running = False
        if self.telemetry_enabled and self._last_telemetry_line_length:
            print()
            self._last_telemetry_line_length = 0
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("停止键盘监听器失败 / Failed to stop the keyboard listener")

        # Returning to zero is opt-in. Never attempt it after a communication
        # or control exception, because moving after a fault may be unsafe.
        if normal_exit and self.return_zero_on_exit:
            try:
                self._safe_return_to_zero()
            except Exception:
                logger.exception("缓慢回零过程中发生异常 / Error during return-to-zero")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 reBot Arm 102 主臂录制动作，并在 B601-RS 从臂上回放。 / Record leader motions and replay them on a B601-RS follower."
    )
    parser.add_argument("--leader-port", default=CFG_LEADER_PORT)
    parser.add_argument("--leader-id", default=CFG_LEADER_ID)
    parser.add_argument("--leader-baudrate", type=int, default=CFG_LEADER_BAUDRATE)
    parser.add_argument("--follower-port", default=CFG_FOLLOWER_PORT)
    parser.add_argument("--follower-id", default=CFG_FOLLOWER_ID)
    parser.add_argument("--can-adapter", default=CFG_CAN_ADAPTER)
    parser.add_argument("--control-hz", type=float, default=CFG_CONTROL_HZ)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=CFG_MAX_RELATIVE_TARGET_DEG,
        help=(
            "LeRobot 每个控制周期允许的最大相对目标变化量，单位为度。 / Maximum relative target change per control cycle, in degrees. "
            "默认值 0 表示关闭该额外限幅，使实时跟随和回放不会被拆成多次追赶。 / The default 0 disables this extra clamp so live follow and "
            "playback are not split into repeated catch-up steps."
        ),
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        default=not CFG_PLAY_LOOP,
        help="每个动作仅播放一次，不循环播放。 / Play each recording once instead of looping.",
    )
    parser.add_argument(
        "--return-zero-on-exit",
        action="store_true",
        default=CFG_RETURN_ZERO_ON_NORMAL_EXIT,
        help="仅在正常退出时，让所有关节缓慢回到零点。 / On a normal exit, smoothly return all joints to zero.",
    )
    parser.add_argument(
        "--print-actions",
        action="store_true",
        default=CFG_PRINT_ACTIONS,
        help="每秒打印两次当前发送的关节动作坐标。 / Print applied joint actions twice per second.",
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        default=not CFG_TELEMETRY_ENABLED,
        help="关闭电机角度和温度实时显示。 / Disable live motor angle and temperature display.",
    )
    parser.add_argument(
        "--telemetry-hz",
        type=float,
        default=CFG_TELEMETRY_HZ,
        help="电机状态显示刷新频率，默认 2 Hz。 / Telemetry refresh rate; default: 2 Hz.",
    )
    parser.add_argument(
        "--temp-warning",
        type=float,
        default=CFG_TEMP_WARNING_C,
        help="温度警告阈值，默认 70°C，仅显示警告标记。 / Warning threshold; default: 70°C.",
    )
    parser.add_argument(
        "--temp-critical",
        type=float,
        default=CFG_TEMP_CRITICAL_C,
        help="自动停机温度，默认 135°C。 / Automatic shutdown temperature; default: 135°C.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default=CFG_LOG_LEVEL,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _silence_repeated_relative_goal_clamp_warning()

    print(
        "[启动配置 / STARTUP CONFIG] "
        f"leader={args.leader_port}, follower={args.follower_port}, "
        f"control={args.control_hz:.1f} Hz, telemetry="
        f"{'ON' if not args.no_telemetry else 'OFF'}, "
        f"temperature shutdown={args.temp_critical:.1f}°C."
    )

    # Disable LeRobot's additional per-cycle catch-up clamp by default.
    # Motion playback and Ctrl+C return-to-zero are still explicitly interpolated
    # by this script at control_hz, so those paths remain smooth.
    max_relative_target = (
        None if args.max_relative_target <= 0 else args.max_relative_target
    )
    if max_relative_target is None:
        print(
            "[配置 / CONFIG] 已关闭相对目标分次追赶限幅； / Relative-goal catch-up clamp disabled; "
            "每个控制周期将直接执行脚本给出的目标。 / Commands are applied directly each control cycle."
        )
    else:
        print(
            "[配置 / CONFIG] 已启用相对目标限幅： / Relative-goal clamp enabled: "
            f"每个控制周期最多变化 {max_relative_target:.2f} 度。 / Maximum {max_relative_target:.2f} deg per control cycle."
        )

    leader_config = RebotArm102LeaderConfig(
        port=args.leader_port,
        id=args.leader_id,
        baudrate=args.leader_baudrate,
    )
    follower_config = SeeedB601RSFollowerConfig(
        port=args.follower_port,
        id=args.follower_id,
        can_adapter=args.can_adapter,
        max_relative_target=max_relative_target,
        disable_torque_on_disconnect=CFG_DISABLE_TORQUE_ON_DISCONNECT,
    )

    leader = RebotArm102Leader(leader_config)
    follower = SeeedB601RSFollower(follower_config)
    controller: RSDanceRecorder | None = None
    normal_exit = False

    stop_signal_count = 0

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal normal_exit, stop_signal_count
        stop_signal_count += 1
        normal_exit = True

        if controller is None:
            # Ctrl+C during hardware connection cannot safely command a return.
            raise KeyboardInterrupt

        if stop_signal_count == 1:
            print(f"\n[退出 / EXIT] 收到信号 {signum}，正在停止当前控制循环。 / Signal {signum} received; stopping the control loop.")
            # Ctrl+C always requests a graceful return, even when the optional
            # command-line flag was not supplied.
            controller.return_zero_on_exit = True
            controller.running = False
        else:
            # Do not raise from inside shutdown/finally.  Let the return loop
            # notice this flag and proceed directly to disconnect.
            controller.abort_return_to_zero = True
            controller.running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        print(f"[连接 / CONNECT] 正在连接 RS 从臂：{args.follower_port} / Connecting RS follower: {args.follower_port}")
        follower.connect(calibrate=True)
        print(f"[连接 / CONNECT] 正在连接主臂：{args.leader_port} / Connecting leader: {args.leader_port}")
        leader.connect(calibrate=True)

        print(
            "[温度监控 / TEMPERATURE] "
            f"警告阈值={args.temp_warning:.1f}°C，"
            f"自动停机阈值={args.temp_critical:.1f}°C，"
            f"刷新频率={args.telemetry_hz:.1f} Hz。 / "
            f"warning={args.temp_warning:.1f}°C, "
            f"shutdown={args.temp_critical:.1f}°C, "
            f"refresh={args.telemetry_hz:.1f} Hz."
        )

        controller = RSDanceRecorder(
            leader,
            follower,
            control_hz=args.control_hz,
            play_loop=not args.no_loop,
            return_zero_on_exit=args.return_zero_on_exit,
            print_actions=args.print_actions,
            telemetry_enabled=not args.no_telemetry,
            telemetry_hz=args.telemetry_hz,
            temp_warning_c=args.temp_warning,
            temp_critical_c=args.temp_critical,
        )
        controller.run()
        normal_exit = True
        return 0

    except KeyboardInterrupt:
        normal_exit = True
        return 0
    except Exception:
        logger.exception(
            "控制程序因异常停止。如果机械臂运动异常，请立即扶住机械臂并切断电机电源。 / Control stopped due to an error. If motion is abnormal, hold the arm and cut motor power immediately."
        )
        return 1
    finally:
        if controller is not None:
            controller.close(normal_exit=normal_exit)

        if leader.is_connected:
            try:
                leader.disconnect()
            except Exception:
                logger.exception("断开主臂连接失败 / Failed to disconnect the leader")

        if follower.is_connected:
            try:
                follower.disconnect()
            except Exception:
                logger.exception("断开 RS 从臂连接失败 / Failed to disconnect the RS follower")


if __name__ == "__main__":
    sys.exit(main())
