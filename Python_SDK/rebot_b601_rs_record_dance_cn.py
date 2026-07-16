#!/usr/bin/env python3
"""
使用 reBot Arm 102 主臂录制动作，并在 reBot B601-RS 从臂上回放。

交互方式保持与 DM 机械臂录制/播放示例一致：
  q/w/e/r/t：开始/停止录制第 1～5 个动作槽位
  1/2/3/4/5：播放第 1～5 个动作槽位
  s：停止录制或播放，并安全返回实时跟随
  c：清除当前选中的动作槽位
  a：清除全部动作槽位
  f：实时主从跟随模式
  Esc：退出程序

录制数据仅保存在当前程序内存中，与原始示例保持一致。
"""

from __future__ import annotations

import argparse
import bisect
import copy
import logging
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

RECORD_KEYS = {"q": 0, "w": 1, "e": 2, "r": 3, "t": 4}
PLAY_KEYS = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}


class Mode(str, Enum):
    FOLLOW = "follow"
    RECORD = "record"
    SYNC_TO_LEADER = "sync_to_leader"
    TRANSITION = "transition"
    PLAYBACK = "playback"


MODE_CN_LABELS = {
    Mode.FOLLOW: "实时跟随",
    Mode.RECORD: "正在录制",
    Mode.SYNC_TO_LEADER: "安全同步",
    Mode.TRANSITION: "平滑过渡",
    Mode.PLAYBACK: "正在播放",
}


@dataclass
class MotionFrame:
    t: float
    action: dict[str, float]


class RSDanceRecorder:
    NUM_SLOTS = 5

    # Kept close to the original DM demo.
    END_HOLD_TIME_S = 0.15
    LOOP_BLEND_TIME_S = 0.30
    RECORD_FILTER_ALPHA = 0.35
    MIN_RECORD_INTERVAL_S = 0.01
    # Original demo used 0.003 rad; this script records degrees.
    MIN_JOINT_CHANGE_DEG = 0.172
    TRANSITION_TIME_S = 0.60

    # Safe playback -> leader synchronization. The bottom-layer relative-goal
    # clamp is disabled for smooth replay, so mode switching is rate-limited here.
    LEADER_SYNC_ARM_MAX_SPEED_DEG_S = 15.0
    LEADER_SYNC_GRIPPER_MAX_SPEED_DEG_S = 5.0
    LEADER_SYNC_TOLERANCE_DEG = 0.50

    # Graceful Ctrl+C return-to-zero settings.  The actual return duration is
    # automatically extended so that no action-space joint exceeds this speed.
    RETURN_ZERO_ARM_MAX_SPEED_DEG_S = 15.0
    # The configured gripper action is multiplied by 6 before reaching the
    # motor, so keep its action-space return speed lower.
    RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S = 5.0
    RETURN_ZERO_MIN_TIME_S = 3.0
    RETURN_ZERO_SETTLE_TIME_S = 0.30

    def __init__(
        self,
        leader: RebotArm102Leader,
        follower: SeeedB601RSFollower,
        *,
        control_hz: float = 30.0,
        play_loop: bool = True,
        return_zero_on_exit: bool = False,
        print_actions: bool = False,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz 必须大于 0")

        self.leader = leader
        self.follower = follower
        self.control_hz = float(control_hz)
        self.control_period_s = 1.0 / self.control_hz
        self.play_loop = bool(play_loop)
        self.return_zero_on_exit = bool(return_zero_on_exit)
        self.print_actions = bool(print_actions)

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
            raise KeyError(f"动作数据缺少以下关节键：{missing}")
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
                raise ValueError(f"joint_directions[{motor_name!r}] 不能为 0")
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
            print("\n[同步] 正在缓慢同步到主臂当前位置，完成后恢复实时跟随。")
        else:
            self.selected_slot = pending_record_slot
            print(
                f"\n[同步] 动作槽位 {pending_record_slot + 1}：正在缓慢同步到主臂位置，"
                "同步完成后才开始录制。请暂时保持主臂稳定，"
                "直到出现‘开始录制’提示。"
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
            f"\n[录制] 动作槽位 {slot + 1}：开始录制 "
            "（再次按相同录制键或按 s 停止）。"
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

            # Add a short blend back to frame zero for seamless loop playback.
            if self.play_loop and len(frames) >= 2:
                frames.append(
                    MotionFrame(
                        t=frames[-1].t + self.LOOP_BLEND_TIME_S,
                        action=copy.deepcopy(frames[0].action),
                    )
                )

        print(
            f"\n[录制] 动作槽位 {slot + 1}：录制已停止，"
            f"共 {len(frames)} 帧，"
            f"时长 {frames[-1].t if frames else 0.0:.2f} 秒。"
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
            print(f"\n[播放] 动作槽位 {slot + 1} 为空，请先录制动作。")
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
            f"\n[播放] 动作槽位 {slot + 1}：正在平滑移动到起始姿态 "
            f"（预计 {self.TRANSITION_TIME_S:.2f} 秒）。"
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
                print(f"\n[播放] 动作槽位 {self.play_slot + 1}：播放完成。")
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
            print("\n[退出] 检测到 Esc，正在退出程序。")
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
                        f"\n[同步] 动作槽位 {slot + 1}：已取消待录制任务，"
                        "继续安全同步到实时跟随姿态。"
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
                        "\n[同步] 已取消待录制任务，"
                        "继续安全同步到实时跟随姿态。"
                    )
                else:
                    self.mode = Mode.FOLLOW
                    self.play_slot = None
                    print("\n[停止] 已停止当前任务，实时跟随已启用。")

            elif char == "f":
                if self.mode == Mode.RECORD:
                    self._stop_recording_locked(now)
                elif self.mode in (Mode.PLAYBACK, Mode.TRANSITION):
                    self._begin_leader_sync_locked(pending_record_slot=None)
                elif self.mode == Mode.SYNC_TO_LEADER:
                    self.pending_record_slot = None
                    print("\n[同步] 正在继续安全同步到实时跟随姿态。")
                else:
                    self.mode = Mode.FOLLOW
                    self.play_slot = None
                    print("\n[跟随] 已进入实时主从跟随模式。")

            elif char == "c":
                if self.mode == Mode.RECORD and self.record_slot == self.selected_slot:
                    self._stop_recording_locked(now)
                self.motion_slots[self.selected_slot] = []
                print(f"\n[清除] 已清除动作槽位 {self.selected_slot + 1}。")

            elif char == "a":
                if self.mode == Mode.RECORD:
                    self._stop_recording_locked(now)
                self.motion_slots = [[] for _ in range(self.NUM_SLOTS)]
                self.mode = Mode.FOLLOW
                self.play_slot = None
                print("\n[清除] 已清除全部动作槽位。")

        return None

    # ------------------------------------------------------------------
    # Main loop / shutdown
    # ------------------------------------------------------------------
    @staticmethod
    def print_help() -> None:
        print(
            "\n"
            "========== reBot B601-RS 动作录制与跳舞回放 ==========\n"
            " q w e r t：先安全同步到主臂，再录制第 1～5 个动作槽位\n"
            " 1 2 3 4 5：播放第 1～5 个动作槽位\n"
            " s         ：停止录制/播放，并安全返回实时跟随\n"
            " f         ：实时主从跟随\n"
            " c         ：清除当前选中的动作槽位\n"
            " a         ：清除全部动作槽位\n"
            " Esc       ：退出程序（仅指定 --return-zero-on-exit 时回零）\n"
            " Ctrl+C    ：停止当前动作并缓慢安全回到零点\n"
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
                "回零前无法读取机械臂当前位置；"
                "将使用最后一次发送的目标姿态作为回零起点"
            )
            if self.last_sent_action is None:
                print("[退出] 无法获取当前位置，已跳过自动回零。请手动扶住机械臂并断电。")
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
            "[回零] 检测到 Ctrl+C，正在缓慢回到零点，"
            f"预计需要 {duration_s:.1f} 秒 "
            f"（机械臂关节峰值速度 ≤ {self.RETURN_ZERO_ARM_MAX_SPEED_DEG_S:.1f} 度/秒，"
            f"夹爪峰值速度 ≤ {self.RETURN_ZERO_GRIPPER_MAX_SPEED_DEG_S:.1f} 度/秒）。"
        )
        print("[回零] 再次按 Ctrl+C 可立即中止回零并断开电机。")

        next_tick = time.monotonic()
        for i in range(1, steps + 1):
            if self.abort_return_to_zero:
                print("[回零] 检测到第二次 Ctrl+C，已中止回零。")
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
        print("[回零] 已到达零点，正在断开电机。")

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

        while self.running:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.monotonic()
            elif now - next_tick > 5 * self.control_period_s:
                # If communication stalls, do not try to execute a large backlog.
                next_tick = now

            with self._lock:
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
                                "\n[同步] 已与主臂位置对齐；"
                                "已恢复实时跟随。"
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
                                f"\n[播放] 动作槽位 {self.play_slot + 1 if self.play_slot is not None else '?'}：正在播放。"
                            )

                elif mode == Mode.PLAYBACK:
                    action = self._playback_action_locked(now)
                    if action is not None:
                        self._send_action(action)

                if self.print_actions and now - last_print_time >= 0.5:
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
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.exception("停止键盘监听器失败")

        # Returning to zero is opt-in. Never attempt it after a communication
        # or control exception, because moving after a fault may be unsafe.
        if normal_exit and self.return_zero_on_exit:
            try:
                self._safe_return_to_zero()
            except Exception:
                logger.exception("缓慢回零过程中发生异常")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 reBot Arm 102 主臂录制动作，并在 B601-RS 从臂上回放。"
    )
    parser.add_argument("--leader-port", default="/dev/ttyUSB0")
    parser.add_argument("--leader-id", default="rebot_arm_102_leader")
    parser.add_argument("--leader-baudrate", type=int, default=1_000_000)
    parser.add_argument("--follower-port", default="can0")
    parser.add_argument("--follower-id", default="follower1")
    parser.add_argument("--can-adapter", default="socketcan")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=0.0,
        help=(
            "LeRobot 每个控制周期允许的最大相对目标变化量，单位为度。"
            "默认值 0 表示关闭该额外限幅，使实时跟随和"
            "回放动作不会被拆分成多次追赶。"
        ),
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="每个动作仅播放一次，不循环播放。",
    )
    parser.add_argument(
        "--return-zero-on-exit",
        action="store_true",
        help="仅在正常退出时，让所有关节缓慢回到零点。",
    )
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="每秒打印两次当前发送的关节动作坐标。",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _silence_repeated_relative_goal_clamp_warning()

    # Disable LeRobot's additional per-cycle catch-up clamp by default.
    # Motion playback and Ctrl+C return-to-zero are still explicitly interpolated
    # by this script at control_hz, so those paths remain smooth.
    max_relative_target = (
        None if args.max_relative_target <= 0 else args.max_relative_target
    )
    if max_relative_target is None:
        print(
            "[配置] 已关闭相对目标分次追赶限幅；"
            "每个控制周期将直接执行脚本给出的目标。"
        )
    else:
        print(
            "[配置] 已启用相对目标限幅："
            f"每个控制周期最多变化 {max_relative_target:.2f} 度。"
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
        disable_torque_on_disconnect=True,
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
            print(f"\n[退出] 收到信号 {signum}，正在停止当前控制循环。")
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
        print(f"[连接] 正在连接 RS 从臂：{args.follower_port}")
        follower.connect(calibrate=True)
        print(f"[连接] 正在连接主臂：{args.leader_port}")
        leader.connect(calibrate=True)

        controller = RSDanceRecorder(
            leader,
            follower,
            control_hz=args.control_hz,
            play_loop=not args.no_loop,
            return_zero_on_exit=args.return_zero_on_exit,
            print_actions=args.print_actions,
        )
        controller.run()
        normal_exit = True
        return 0

    except KeyboardInterrupt:
        normal_exit = True
        return 0
    except Exception:
        logger.exception(
            "控制程序因异常停止。如果机械臂运动异常，请立即扶住机械臂并切断电机电源。"
        )
        return 1
    finally:
        if controller is not None:
            controller.close(normal_exit=normal_exit)

        if leader.is_connected:
            try:
                leader.disconnect()
            except Exception:
                logger.exception("断开主臂连接失败")

        if follower.is_connected:
            try:
                follower.disconnect()
            except Exception:
                logger.exception("断开 RS 从臂连接失败")


if __name__ == "__main__":
    sys.exit(main())