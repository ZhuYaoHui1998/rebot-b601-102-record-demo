
* Python 语法编译检查
* 录制、停止、过渡、插值、循环播放的模拟测试
* RS 关节方向与夹爪 6 倍比例的转换检查

由于我无法连接你的实体机械臂，实际关节方向、零位和夹爪力度仍需你在低速、空载状态下验证。

## 一、保留了哪些原版逻辑

我保留了第一个 DM 项目的核心交互方式：

| 按键                  | 功能               |
| ------------------- | ---------------- |
| `q / w / e / r / t` | 录制或停止录制第 1～5 槽动作 |
| `1 / 2 / 3 / 4 / 5` | 播放第 1～5 槽动作      |
| `s`                 | 停止录制或播放，回到实时跟随   |
| `f`                 | 回到主从实时跟随         |
| `c`                 | 清除当前选中的槽位        |
| `a`                 | 清除所有槽位           |
| `Esc` 或 `Ctrl+C`    | 退出               |

同时保留了：

* 5 个内存动作槽
* 录制过程低通滤波
* 最小动作变化阈值
* 播放前平滑移动到起始姿态
* 帧间平滑插值
* 结束姿态短暂停留
* 末尾平滑连接第一帧
* 自动循环跳舞

这些都是按照第一个 DM 示例中的录制和播放结构重新实现的。([GitHub][1])

## 二、RS 版本与 DM 版本的区别

没有继续使用原 DM 项目中的底层电机发送代码，而是换成：

```python
RebotArm102Leader.get_action()
```

读取主臂，以及：

```python
SeeedB601RSFollower.send_action()
```

控制 RS 从臂。

官方 Leader 返回的动作和 RS Follower 接收的动作使用同一套关节名称：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_yaw.pos
wrist_roll.pos
gripper.pos
```

Leader 输出单位是度，Follower 会在 `send_action()` 内部处理关节方向、软限位、弧度转换和夹爪比例，因此录制的数据可以直接用于 RS 回放。([GitHub][2])

代码还专门处理了一个容易出错的地方：Follower 观测值是电机实际坐标，而录制值是 Leader 动作坐标。因此播放前读取当前姿态时，会反向除以 `joint_directions`。否则肘关节、腕关节和夹爪可能被二次反向或二次放大。

## 三、先确认两个软件包能导入

在你运行 LeRobot 的同一个虚拟环境中执行：

```bash
python - <<'PY'
from lerobot_robot_seeed_b601 import (
    SeeedB601RSFollower,
    SeeedB601RSFollowerConfig,
)

from lerobot_teleoperator_rebot_arm_102 import (
    RebotArm102Leader,
    RebotArm102LeaderConfig,
)

print("RS follower and leader imports OK")
PY
```

出现：

```text
RS follower and leader imports OK
```

说明环境正确。

如果缺少 `pynput`：

```bash
pip install pynput
```

如果键盘按键没有反应，Seeed 教程建议尝试：

```bash
pip install pynput==1.6.8
```

([Seeed Studio][3])

## 四、启动 CAN 和串口权限

```bash
sudo chmod 666 /dev/ttyUSB0

sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
```

查看状态：

```bash
ip -details link show can0
```

应看到类似：

```text
state UP
bitrate 1000000
```

这些端口和 CAN 配置与 Seeed 官方 RS 遥操作教程一致。([Seeed Studio][3])

## 五、运行代码

把下载的代码放入你的工作目录，然后执行：

```bash
python rebot_b601_rs_record_dance.py \
    --leader-port /dev/ttyUSB0 \
    --follower-port can0
```

如果你的主臂串口不是 `/dev/ttyUSB0`，先查看：

```bash
ls /dev/ttyUSB*
```

例如是 `/dev/ttyUSB1`：

```bash
python rebot_b601_rs_record_dance.py \
    --leader-port /dev/ttyUSB1 \
    --follower-port can0
```

## 六、操作示例

### 录制第一段舞蹈

启动程序后：

1. 按 `q` 开始录制槽位1。
2. 用主臂带动 RS 从臂运动。
3. 再按一次 `q`，或者按 `s`，停止录制。
4. 按数字 `1`，播放槽位1。
5. 默认会循环播放。

录制第二段：

```text
按 w → 移动主臂 → 再按 w → 按 2 播放
```

以此类推：

```text
q 对应槽位1
w 对应槽位2
e 对应槽位3
r 对应槽位4
t 对应槽位5
```


标准遥操作正常后，再运行跳舞代码。第一次测试建议：

1. 不夹物体。
2. 机械臂周围清空。
3. 动作幅度保持较小。
4. 手放在急停或电源开关附近。
5. 先录制一个只有一两个关节缓慢运动的动作。
6. 确认播放方向正确后，再录制完整舞蹈。

你前面修改的 RS 夹爪力矩保护仍会生效，因为这个脚本最终仍调用 `SeeedB601RSFollower.send_action()`，RS 夹爪命令会继续进入你修改后的 `mit_output_torque_limit()`。

[1]: https://github.com/Welt-liu/rebot-b601-102-record-demo.git "GitHub - Welt-liu/rebot-b601-102-record-demo · GitHub"
[2]: https://github.com/Seeed-Projects/lerobot-teleoperator-rebot-arm-102/raw/refs/heads/main/lerobot_teleoperator_rebot_arm_102/rebot_arm_102_leader.py "raw.githubusercontent.com"
[3]: https://wiki.seeedstudio.com/cn/rebot_arm_b601_rs_lerobot/ "reBot Arm B601-RS入门Lerobot | Seeed Studio Wiki"
[4]: https://github.com/Seeed-Projects/lerobot-robot-seeed-b601/raw/refs/heads/main/lerobot_robot_seeed_b601/seeed_b601_follower.py "raw.githubusercontent.com"

