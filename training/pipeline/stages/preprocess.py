"""训练流水线 stage 占位。

真实实现接入时，该模块应只处理本 stage 的输入/输出，不绑定边缘端内部路径。
"""

from __future__ import annotations


def run(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError("该 stage 尚未接入真实训练流水线")
