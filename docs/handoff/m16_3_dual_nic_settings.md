# M16.3 双网口配置接入

## 目标

在 M16 视觉盒子设置页中接入 eth0 / eth1 双网口配置：

- 自动读取 eth0、eth1 当前 IP、子网掩码、网关、MAC、链路状态。
- Web 页面允许修改 IP / 子网掩码 / 网关。
- 保存视觉盒子设置时，若网络参数与当前系统状态不同，立即调用 `ip` 命令应用。
- 配置仍保存到 `/opt/visionops_v3/config/vision_box_settings.json`。

## 接口

继续使用：

- `GET /api/settings/vision_box`
- `POST /api/settings/vision_box`

`GET` 返回中新增：

```json
"network": {
  "mode": "dual_nic_static",
  "items": [
    {"interface":"eth0", "ip":"...", "netmask":"...", "gateway":"..."},
    {"interface":"eth1", "ip":"...", "netmask":"...", "gateway":"..."}
  ],
  "interfaces": {"eth0": {}, "eth1": {}},
  "configured": {}
}
```

`POST` 可提交：

```json
"network": {
  "interfaces": {
    "eth0": {"ip":"192.168.1.121", "netmask":"255.255.255.0", "gateway":"192.168.1.1"},
    "eth1": {"ip":"192.168.2.121", "netmask":"255.255.255.0", "gateway":""}
  }
}
```

## 应用逻辑

后端先比较目标网络配置和当前系统实时状态：

- 未变化：只保存必要配置，不调用 `ip` 命令。
- 有变化：执行：
  - `ip link set dev <iface> up`
  - `ip -4 addr flush dev <iface> scope global`
  - `ip addr add <ip>/<prefix> dev <iface>`
  - `ip route replace default via <gateway> dev <iface> metric <metric>`

默认 metric：

- eth0: 100
- eth1: 200

如果 gateway 为空，会尝试删除该网口默认路由；失败会忽略。

## 注意

当前实现是立即生效配置，配置会保存到 vision box settings JSON。若需要开机后自动恢复，可后续增加一个 systemd oneshot 服务，在 boot 时读取该 JSON 并调用相同逻辑应用。

修改网络可能导致 Web 连接短暂中断，尤其是修改当前访问 Web 所走的网口时。
