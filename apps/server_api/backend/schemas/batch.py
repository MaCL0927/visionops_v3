"""服务端数据批次字段说明。"""

# extracted: 已从 incoming tar.gz 解压，尚未在第二步确认任务类型。
# accepted: 标注/审核人员已确认，可用于构建 dataset。
# rejected: 不进入训练数据集。
# failed: 上传包处理失败。
BATCH_STATUSES = ("extracted", "accepted", "rejected", "failed")
