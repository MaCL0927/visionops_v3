const state = {
  health: null,
  incomingRoot: '',
  incomingPackages: [],
  batches: [],
  datasets: [],
  jobs: [],
  models: [],
  devices: [],
  selectedPackageNames: new Set(),
  activeLogJobId: '',
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.message || data.detail || res.statusText);
  return data;
}

function $(id) { return document.getElementById(id); }
function pretty(value) { return JSON.stringify(value, null, 2); }
function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}
function formatTime(ms) {
  if (!ms) return '-';
  try { return new Date(Number(ms)).toLocaleString(); } catch { return '-'; }
}
function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (!Number.isFinite(value) || value <= 0) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let current = value;
  let index = 0;
  while (current >= 1024 && index < units.length - 1) { current /= 1024; index += 1; }
  return `${current.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
function formatPercent(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(1)}%` : '-';
}
function badgeClass(status) {
  if (['ok', 'ready', 'success', 'accepted', 'connect', 'connected', 'synced'].includes(status)) return '';
  if (['failed', 'rejected', 'error', 'fail', 'sync_failed'].includes(status)) return 'danger';
  if (['extracted', 'uploaded', 'pending', 'running', 'assigned', 'unassigned'].includes(status)) return 'warning';
  return 'muted';
}
function showModal(title, body) {
  $('modalTitle').textContent = title;
  $('modalBody').textContent = typeof body === 'string' ? body : pretty(body);
  $('modal').style.display = 'flex';
}
function showToast(message, kind = 'success') {
  let root = document.querySelector('.toast-root');
  if (!root) {
    root = document.createElement('div');
    root.className = 'toast-root';
    document.body.appendChild(root);
  }
  const item = document.createElement('div');
  item.className = 'toast ' + (kind || 'success');
  item.textContent = String(message || '操作完成');
  root.appendChild(item);
  setTimeout(() => { item.style.opacity = '0'; item.style.transform = 'translateY(8px)'; }, 2600);
  setTimeout(() => item.remove(), 3100);
}

function latestJob() {
  const jobs = [...state.jobs].sort((a, b) => Number(b.updated_at_ms || b.created_at_ms || 0) - Number(a.updated_at_ms || a.created_at_ms || 0));
  return jobs.find(x => x.status === 'running') || jobs[0] || null;
}

async function refresh() {
  const [health, incoming, batches, datasets, jobs, models, devices] = await Promise.all([
    api('/api/server/health'),
    api('/api/server/incoming-packages'),
    api('/api/server/batches'),
    api('/api/server/datasets'),
    api('/api/server/training/jobs'),
    api('/api/server/model-packages'),
    api('/api/server/devices'),
  ]);
  state.health = health;
  state.incomingRoot = incoming.incoming_root || health.incoming_root || '';
  state.incomingPackages = incoming.packages || [];
  state.batches = batches.batches || [];
  state.datasets = datasets.datasets || [];
  state.jobs = jobs.jobs || [];
  state.models = models.model_packages || [];
  state.devices = devices.devices || [];
  state.selectedPackageNames = new Set([...state.selectedPackageNames].filter(name => state.incomingPackages.some(x => x.name === name)));
  renderAll();
  await updateRealtimeLog(false);
}

function renderAll() {
  renderStatus();
  renderIncomingPackages();
  renderBatches();
  renderDatasets();
  renderJobs();
  renderModels();
  renderDevices();
  refreshSelects();
}

function renderStatus() {
  const h = state.health || {};
  const badge = $('serverBadge');
  if (badge) {
    badge.textContent = h.status === 'ok' ? '服务端：已连接' : '服务端：未连接';
    badge.className = 'badge ' + (h.status === 'ok' ? '' : 'danger');
  }
  if ($('mlflowLink')) $('mlflowLink').href = h.mlflow_uri || '#';
  renderSystemStats(h.system_stats || {});
  if ($('publishRootInput') && !$('publishRootInput').dataset.touched && !$('publishRootInput').value && h.publish_root) {
    $('publishRootInput').placeholder = h.publish_root;
  }
}

function renderSystemStats(stats) {
  const root = $('systemStats');
  if (!root) return;
  const disk = stats.disk || {};
  const memory = stats.memory || {};
  const cpu = stats.cpu || {};
  const gpu = stats.gpu || {};
  const gpuText = gpu.available ? formatPercent(gpu.percent) : 'N/A';
  root.innerHTML = `
    <span title="visionops_v3 目录占用">VisionOps ${formatBytes(stats.visionops_size_bytes)}</span>
    <span title="当前分区磁盘使用率">磁盘 ${formatPercent(disk.percent)}</span>
    <span title="内存使用率">内存 ${formatPercent(memory.percent)}</span>
    <span title="CPU 使用率">CPU ${formatPercent(cpu.percent)}</span>
    <span title="GPU 使用率">GPU ${gpuText}</span>
  `;
}

function renderIncomingPackages() {
  $('incomingRootText').textContent = state.incomingRoot || '-';
  const root = $('incomingPackages');
  if (!state.incomingPackages.length) {
    root.className = 'list empty incoming-list';
    root.textContent = `当前没有待处理上传包。目录：${state.incomingRoot || '-'}`;
    return;
  }
  root.className = 'list incoming-list';
  root.innerHTML = state.incomingPackages.map(item => `
    <div class="item">
      <div class="item-main">
        <input type="checkbox" class="pkg-check" data-name="${escapeHtml(item.name)}" ${state.selectedPackageNames.has(item.name) ? 'checked' : ''} />
        <div>
          <div class="item-title">${escapeHtml(item.name)} <span class="badge warning">pending</span></div>
          <div class="item-meta">device=${escapeHtml(item.device_id)} | customer=${escapeHtml(item.customer_id)} | captured=${escapeHtml(item.captured_at)} | ${item.size_mb || 0} MB | ${formatTime(item.mtime_ms)}</div>
        </div>
        <div class="item-actions">
          <button onclick="showModal('上传包信息', state.incomingPackages.find(x => x.name === '${escapeHtml(item.name)}'))" class="secondary">详情</button>
        </div>
      </div>
    </div>`).join('');
  document.querySelectorAll('.pkg-check').forEach(el => {
    el.onchange = (event) => {
      const name = event.target.dataset.name;
      if (event.target.checked) state.selectedPackageNames.add(name); else state.selectedPackageNames.delete(name);
    };
  });
}

function renderBatches() {
  const root = $('batches');
  if (!root) return;
  const items = [...state.batches].sort((a, b) => Number(b.created_at_ms || b.updated_at_ms || 0) - Number(a.created_at_ms || a.updated_at_ms || 0));
  if (!items.length) {
    root.className = 'folder-list empty';
    root.textContent = '暂无已解压数据文件夹。请先在第 1 步处理 incoming 上传包。';
    return;
  }
  root.className = 'folder-list';
  root.innerHTML = items.map(item => `
    <div class="folder-card">
      <div class="folder-icon">📁</div>
      <div>
        <div class="folder-title">${escapeHtml(item.batch_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status || 'extracted')}</span></div>
        <div class="folder-meta">device=${escapeHtml(item.device_id || '-')} | customer=${escapeHtml(item.customer_id || '-')} | task=${escapeHtml(item.task_type || 'unassigned')} | images=${item.image_count || 0} | labels=${item.label_count || 0} | ${formatTime(item.created_at_ms)}</div>
      </div>
      <div class="folder-actions">
        <button onclick="openAnnotator('${escapeHtml(item.batch_id)}')">标注</button>
        <button onclick="openBatchFolder('${escapeHtml(item.batch_id)}')" class="secondary">详情</button>
        <button onclick="deleteBatch('${escapeHtml(item.batch_id)}')" class="danger">删除</button>
      </div>
    </div>`).join('');
}

function renderDatasets() {
  const root = $('datasets');
  if (!root) return;
  const items = [...state.datasets].sort((a, b) => Number(b.created_at_ms || 0) - Number(a.created_at_ms || 0));
  if (!items.length) {
    root.className = 'list empty compact-list';
    root.textContent = '暂无 dataset。标注器点击“确认审核完成”后会自动生成。';
    return;
  }
  root.className = 'list compact-list';
  root.innerHTML = items.map(item => `
    <div class="item">
      <div class="item-title">${escapeHtml(item.dataset_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status || 'ready')}</span></div>
      <div class="item-meta">${escapeHtml(item.task_type)} | batches=${(item.batch_ids || []).length} | images=${item.image_count || 0} | labels=${item.label_count || 0} | classes=${item.class_count || 0}</div>
      <div class="item-actions">
        <button onclick="openDatasetFolder('${escapeHtml(item.dataset_id)}')" class="secondary">详情</button>
        <button onclick="deleteDataset('${escapeHtml(item.dataset_id)}')" class="danger">删除</button>
      </div>
    </div>`).join('');
}

function renderJobs() {
  const root = $('jobs');
  if (!root) return;
  const items = [...state.jobs].sort((a, b) => Number(b.updated_at_ms || b.created_at_ms || 0) - Number(a.updated_at_ms || a.created_at_ms || 0));
  if (!items.length) { root.className = 'list empty training-job-list'; root.textContent = '暂无 training job。'; return; }
  root.className = 'list training-job-list compact-row-list';
  root.innerHTML = items.map(item => {
    const canCancel = !['success', 'failed', 'canceled'].includes(String(item.status || ''));
    const cancelBtn = canCancel ? `<button onclick="cancelJob('${escapeHtml(item.job_id)}')" class="warning">取消</button>` : '';
    return `
    <div class="item item-compact">
      <div class="inline-info">
        <span class="item-title">${escapeHtml(item.job_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status)}</span></span>
        <span class="item-meta inline-meta">stage=${escapeHtml(item.current_stage)} | dataset=${escapeHtml(item.dataset_id)} | model=${escapeHtml(item.output_model_package || '-')} | ${formatTime(item.updated_at_ms)}</span>
      </div>
      <div class="item-actions">
        ${cancelBtn}
        <button onclick="openJobFolder('${escapeHtml(item.job_id)}')" class="secondary">详情</button>
        <button onclick="deleteJob('${escapeHtml(item.job_id)}')" class="danger">删除</button>
      </div>
    </div>`;
  }).join('');
}

function renderModels() {
  const root = $('models');
  if (!state.models.length) { root.className = 'list empty'; root.textContent = '暂无模型包。训练任务成功后会自动生成。'; return; }
  root.className = 'list compact-row-list model-package-list';
  const items = [...state.models].sort((a, b) => Number(b.updated_at_ms || b.created_at_ms || 0) - Number(a.updated_at_ms || a.created_at_ms || 0));
  root.innerHTML = items.map(item => `
    <div class="item item-compact">
      <div class="inline-info">
        <span class="item-title">${escapeHtml(item.model_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status)}</span></span>
        <span class="item-meta inline-meta">${escapeHtml(item.task_type)} | platform=${escapeHtml(item.target_platform)} | job=${escapeHtml(item.job_id || '-')}</span>
      </div>
      <div class="item-actions">
        <button onclick="publishModel('${escapeHtml(item.model_id)}')" class="success">发布</button>
        <button onclick="openModelFolder('${escapeHtml(item.model_id)}')" class="secondary">详情</button>
        <button onclick="deleteModel('${escapeHtml(item.model_id)}')" class="danger">删除</button>
      </div>
    </div>`).join('');
}

function renderDevices() {
  const root = $('devices');
  if (!state.devices.length) { root.className = 'list empty'; root.textContent = '暂无设备。'; return; }
  root.className = 'list compact-row-list device-list';
  root.innerHTML = state.devices.map(item => {
    const deviceUser = item.device_user || item.ssh_user || item.user || 'neardi';
    const collectorStatus = item.collector_status || 'unknown';
    return `
    <div class="item item-compact">
      <div class="inline-info">
        <span class="item-title">${escapeHtml(item.device_id)} <span class="badge ${badgeClass(item.sync_status)}">${escapeHtml(item.sync_status || 'unknown')}</span><span class="badge ${badgeClass(collectorStatus)}">${escapeHtml(collectorStatus)}</span></span>
        <span class="item-meta inline-meta">${escapeHtml(deviceUser)} | ip=${escapeHtml(item.ip || '-')} | root=${escapeHtml(item.model_root || '-')} | current=${escapeHtml(item.current_model || '-')} | target=${escapeHtml(item.target_model || '-')}</span>
      </div>
      <div class="item-actions">
        <button onclick="showModal('device 详情', state.devices.find(x => x.device_id === '${escapeHtml(item.device_id)}'))" class="secondary">详情</button>
        <button onclick="deleteDevice('${escapeHtml(item.device_id)}')" class="danger">删除</button>
      </div>
    </div>`;
  }).join('');
}

function refreshSelects() {
  const datasetSelect = $('datasetSelect');
  const previousDataset = datasetSelect ? datasetSelect.value : '';
  const datasets = [...state.datasets].sort((a, b) => Number(b.created_at_ms || 0) - Number(a.created_at_ms || 0));
  if (datasetSelect) {
    datasetSelect.innerHTML = datasets.length
      ? datasets.map(d => `<option value="${escapeHtml(d.dataset_id)}" data-task="${escapeHtml(d.task_type)}">${escapeHtml(d.dataset_id)} (${escapeHtml(d.task_type)})</option>`).join('')
      : '<option value="">请先完成标注审核生成 dataset</option>';
    datasetSelect.disabled = !datasets.length;
    if (previousDataset && datasets.some(d => d.dataset_id === previousDataset)) datasetSelect.value = previousDataset;
  }

  const deviceSelect = $('assignDeviceSelect');
  const previousDevice = deviceSelect ? deviceSelect.value : '';
  if (deviceSelect) {
    deviceSelect.innerHTML = state.devices.length
      ? state.devices.map(d => {
          const status = d.collector_status || 'unknown';
          return `<option value="${escapeHtml(d.device_id)}">${escapeHtml(d.device_id)} (${escapeHtml(status)})</option>`;
        }).join('')
      : '<option value="">请先登记设备</option>';
    deviceSelect.disabled = !state.devices.length;
    if (previousDevice && state.devices.some(d => d.device_id === previousDevice)) deviceSelect.value = previousDevice;
  }

  const modelSelect = $('assignModelSelect');
  const previousModel = modelSelect ? modelSelect.value : '';
  if (modelSelect) {
    modelSelect.innerHTML = state.models.length
      ? state.models.map(m => `<option value="${escapeHtml(m.model_id)}">${escapeHtml(m.model_id)}</option>`).join('')
      : '<option value="">请先生成模型包</option>';
    modelSelect.disabled = !state.models.length;
    if (previousModel && state.models.some(m => m.model_id === previousModel)) modelSelect.value = previousModel;
  }
}

function taskFromDatasetId(datasetId) {
  const dataset = state.datasets.find(x => x.dataset_id === datasetId);
  return dataset ? dataset.task_type : 'detection';
}

function openAnnotator(batchId) {
  if (!batchId) return showToast('batch_id 为空', 'error');
  window.location.href = '/annotate?batch_id=' + encodeURIComponent(batchId);
}

async function openPath(path) {
  if (!path) return showToast('路径为空', 'error');
  await api('/api/server/open-path', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({path})});
  showToast('已请求打开文件夹：' + path, 'success');
}
async function openBatchFolder(batchId) {
  const batch = state.batches.find(x => x.batch_id === batchId);
  if (!batch) return showToast('batch 不存在', 'error');
  await openPath(batch.raw_path || batch.batch_path);
}
async function openDatasetFolder(datasetId) {
  const dataset = state.datasets.find(x => x.dataset_id === datasetId);
  if (!dataset) return showToast('dataset 不存在', 'error');
  await openPath(dataset.dataset_path);
}
async function openJobFolder(jobId) {
  const job = state.jobs.find(x => x.job_id === jobId);
  if (!job) return showToast('training job 不存在', 'error');
  await openPath(job.job_path);
}
async function openModelFolder(modelId) {
  const model = state.models.find(x => x.model_id === modelId);
  if (!model) return showToast('模型包不存在', 'error');
  await openPath(model.package_path);
}
async function openPublishRoot() {
  const manual = $('publishRootInput') ? $('publishRootInput').value.trim() : '';
  const configured = state.health && state.health.publish_root ? state.health.publish_root : '';
  const target = manual || configured;
  if (!target) return showToast('publish_root 为空，请先在启动参数或输入框中设置发布目录。', 'error');
  await openPath(target);
}

async function processSelectedPackages() {
  const packages = [...state.selectedPackageNames];
  if (!packages.length) return showToast('请先勾选 incoming 目录下的 tar.gz 上传包。');
  const result = await api('/api/server/batches/process-incoming', {
    method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({packages})
  });
  state.selectedPackageNames.clear();
  showToast('上传包处理完成，已生成标注数据文件夹：' + (result.batch && result.batch.batch_id ? result.batch.batch_id : ''), 'success');
  await refresh();
}

async function deleteBatch(batchId) {
  if (!batchId) return;
  await api(`/api/server/batches/${encodeURIComponent(batchId)}/delete`, {method:'POST', body:'{}'});
  showToast('已删除数据文件夹：' + batchId, 'success');
  await refresh();
}
async function deleteDataset(datasetId) {
  if (!datasetId) return;
  await api(`/api/server/datasets/${encodeURIComponent(datasetId)}/delete`, {method:'POST', body:'{}'});
  showToast('已删除数据集：' + datasetId, 'success');
  await refresh();
}
async function deleteJob(jobId) {
  if (!jobId) return;
  await api(`/api/server/training/jobs/${encodeURIComponent(jobId)}/delete`, {method:'POST', body:'{}'});
  if (state.activeLogJobId === jobId) state.activeLogJobId = '';
  showToast('已删除训练任务：' + jobId, 'success');
  await refresh();
}
async function deleteModel(modelId) {
  if (!modelId) return;
  await api(`/api/server/model-packages/${encodeURIComponent(modelId)}/delete`, {method:'POST', body:'{}'});
  showToast('已删除模型包：' + modelId, 'success');
  await refresh();
}
async function deleteDevice(deviceId) {
  if (!deviceId) return showToast('请先选择设备。', 'error');
  await api(`/api/server/devices/${encodeURIComponent(deviceId)}/delete`, {method:'POST', body:'{}'});
  showToast('已删除设备：' + deviceId, 'success');
  await refresh();
}

async function setActiveLogJob(jobId) {
  state.activeLogJobId = jobId || '';
  await updateRealtimeLog(false);
}
async function updateRealtimeLog(showToastOnEmpty=false) {
  const job = state.activeLogJobId ? state.jobs.find(x => x.job_id === state.activeLogJobId) : latestJob();
  if (!job) {
    $('jobLog').textContent = '暂无任务日志。';
    if (showToastOnEmpty) showToast('暂无训练任务日志');
    return;
  }
  state.activeLogJobId = job.job_id;
  const result = await api(`/api/server/training/jobs/${encodeURIComponent(job.job_id)}/logs`);
  const header = `[job] ${job.job_id}\n[status] ${job.status} | stage=${job.current_stage} | dataset=${job.dataset_id}\n\n`;
  $('jobLog').textContent = header + (result.logs || '暂无日志。');
  const pre = $('jobLog');
  pre.scrollTop = pre.scrollHeight;
}

async function cancelJob(jobId) {
  await api(`/api/server/training/jobs/${encodeURIComponent(jobId)}/cancel`, {method:'POST', body:'{}'});
  await refresh();
}
async function publishModel(modelId) {
  const publishRoot = $('publishRootInput').value.trim();
  const body = publishRoot ? {publish_root: publishRoot} : {};
  const result = await api(`/api/server/model-packages/${encodeURIComponent(modelId)}/publish`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const publishPath = result.publish && result.publish.publish_path ? result.publish.publish_path : '';
  showToast('模型包已发布到：' + publishPath, 'success');
  await refresh();
}

$('refreshBtn').onclick = () => refresh().catch(err => showToast(err.message, 'error'));
if ($('modalClose')) $('modalClose').onclick = () => { $('modal').style.display = 'none'; };
if ($('modal')) $('modal').onclick = (event) => { if (event.target.id === 'modal') $('modal').style.display = 'none'; };
if ($('refreshIncomingBtn')) $('refreshIncomingBtn').onclick = () => refresh().catch(err => showToast(err.message, 'error'));
if ($('processIncomingBtn')) $('processIncomingBtn').onclick = () => processSelectedPackages().catch(err => showToast(err.message, 'error'));
if ($('publishRootInput')) $('publishRootInput').addEventListener('input', () => { $('publishRootInput').dataset.touched = '1'; });
if ($('openPublishRootBtn')) $('openPublishRootBtn').onclick = () => openPublishRoot().catch(err => showToast(err.message, 'error'));

if ($('jobForm')) $('jobForm').onsubmit = async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const datasetId = String(form.get('dataset_id') || '').trim();
  if (!datasetId) return showToast('请先完成标注审核生成 dataset。', 'error');
  const body = {
    dataset_id: datasetId,
    task_type: taskFromDatasetId(datasetId),
    epochs: Number(form.get('epochs') || 100),
    batch_size: Number(form.get('batch_size') || 4),
    imgsz: Number(form.get('imgsz') || 640),
    device: '0',
    target_platform: String(form.get('target_platform') || 'rk3576').trim() || 'rk3576',
    conda_executable: 'conda',
    onnx_conda_env: 'pt2onnx',
    rknn_conda_env: 'rknn311',
    amp: form.get('amp') === 'on',
    do_quantization: form.get('do_quantization') === 'on',
  };
  const result = await api('/api/server/training/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  state.activeLogJobId = result.job.job_id;
  showToast('训练任务已创建：' + result.job.job_id, 'success');
  await updateRealtimeLog(false).catch(() => {});
  setTimeout(() => refresh().catch(console.error), 200);
  setTimeout(() => refresh().catch(console.error), 1000);
};

if ($('deviceForm')) $('deviceForm').onsubmit = async (event) => {
  event.preventDefault();
  const body = Object.fromEntries(new FormData(event.target).entries());
  const result = await api('/api/server/devices', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const status = result.device && result.device.collector_status ? result.device.collector_status : 'unknown';
  showToast('设备已更新，SSH状态：' + status, status === 'connect' ? 'success' : (status === 'fail' ? 'error' : 'success'));
  await refresh();
};
if ($('assignModelBtn')) $('assignModelBtn').onclick = async () => {
  const deviceId = $('assignDeviceSelect').value;
  const modelId = $('assignModelSelect').value;
  if (!deviceId || !modelId) return showToast('请先登记设备并生成模型包。');
  const result = await api(`/api/server/devices/${encodeURIComponent(deviceId)}/assign-model`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model_id: modelId})});
  const remoteDir = result.sync && result.sync.remote_dir ? result.sync.remote_dir : '';
  showToast(`目标模型已通过 SSH 同步到 ${deviceId}${remoteDir ? '：' + remoteDir : ''}`, 'success');
  await refresh();
};
refresh().catch(err => {
  if ($('serverBadge')) {
    $('serverBadge').textContent = '服务端：连接失败';
    $('serverBadge').className = 'badge danger';
  }
  $('jobLog').textContent = err.message;
});
setInterval(() => {
  refresh().catch(() => {});
}, 3000);
setInterval(() => {
  updateRealtimeLog(false).catch(() => {});
}, 1200);
