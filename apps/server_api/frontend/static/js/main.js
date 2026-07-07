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
  selectedBatchIds: new Set(),
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
function badgeClass(status) {
  if (['ok', 'ready', 'success', 'accepted'].includes(status)) return '';
  if (['failed', 'rejected', 'error'].includes(status)) return 'danger';
  if (['extracted', 'uploaded', 'pending', 'running', 'assigned', 'unassigned'].includes(status)) return 'warning';
  return 'muted';
}
function showModal(title, body) {
  $('modalTitle').textContent = title;
  $('modalBody').textContent = typeof body === 'string' ? body : pretty(body);
  $('modal').style.display = 'flex';
}
function showToast(message) { showModal('提示', message); }

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
  state.selectedBatchIds = new Set([...state.selectedBatchIds].filter(id => state.batches.some(x => x.batch_id === id)));
  renderAll();
}

function renderAll() {
  renderOverview();
  renderIncomingPackages();
  renderBatches();
  renderDatasets();
  renderJobs();
  renderModels();
  renderDevices();
  refreshSelects();
}

function renderOverview() {
  const h = state.health || {};
  $('serverBadge').textContent = h.status === 'ok' ? '服务端：已连接' : '服务端：未连接';
  $('serverBadge').className = 'badge ' + (h.status === 'ok' ? '' : 'danger');
  $('mlflowLink').href = h.mlflow_uri || '#';
  $('overviewCards').innerHTML = [
    ['待处理包', state.incomingPackages.length],
    ['数据批次', state.batches.length],
    ['已确认批次', state.batches.filter(x => x.status === 'accepted').length],
    ['数据集', state.datasets.length],
    ['训练任务', state.jobs.length],
    ['模型包', state.models.length],
    ['设备', state.devices.length],
  ].map(([name, value]) => `<div class="metric"><div class="value">${value}</div><div class="name">${name}</div></div>`).join('');

  const latestBatch = state.batches.at(-1) || state.batches[0] || null;
  const latestDataset = state.datasets.at(-1) || state.datasets[0] || null;
  const latestJob = state.jobs.at(-1) || state.jobs[0] || null;
  const latestModel = state.models.at(-1) || state.models[0] || null;
  $('currentContext').textContent = pretty({
    data_root: h.data_root,
    incoming_root: state.incomingRoot,
    batch_root: h.batch_root,
    publish_root: h.publish_root || '(未配置，可在模型发布时手动填写)',
    current_batch: latestBatch ? `${latestBatch.batch_id} (${latestBatch.status}, task=${latestBatch.task_type}, images=${latestBatch.image_count}, labels=${latestBatch.label_count})` : null,
    current_dataset: latestDataset ? `${latestDataset.dataset_id} (${latestDataset.task_type}, images=${latestDataset.image_count})` : null,
    latest_job: latestJob ? `${latestJob.job_id} (${latestJob.status}, stage=${latestJob.current_stage})` : null,
    latest_model: latestModel ? `${latestModel.model_id} (${latestModel.status})` : null,
  });
  if (latestJob) loadJobLogs(latestJob.job_id, false).catch(() => {});
}

function renderIncomingPackages() {
  $('incomingRootText').textContent = state.incomingRoot || '-';
  const root = $('incomingPackages');
  if (!state.incomingPackages.length) {
    root.className = 'list empty';
    root.textContent = `当前没有待处理上传包。目录：${state.incomingRoot || '-'}`;
    return;
  }
  root.className = 'list';
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
  const taskType = $('datasetTaskType') ? $('datasetTaskType').value : 'detection';
  if (!state.batches.length) { root.className = 'list empty'; root.textContent = '暂无已解压 batch。请先在第 1 步处理 incoming 上传包。'; return; }
  root.className = 'list';
  root.innerHTML = state.batches.map(item => `
    <div class="item">
      <div class="item-main">
        <input type="checkbox" class="batch-check" data-id="${escapeHtml(item.batch_id)}" ${state.selectedBatchIds.has(item.batch_id) ? 'checked' : ''} />
        <div>
          <div class="item-title">${escapeHtml(item.batch_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status)}</span></div>
          <div class="item-meta">task=${escapeHtml(item.task_type || 'unassigned')} | device=${escapeHtml(item.device_id)} | customer=${escapeHtml(item.customer_id || '-')} | images=${item.image_count || 0} | labels=${item.label_count || 0} | ${formatTime(item.created_at_ms)}</div>
        </div>
        <div class="item-actions">
          <button onclick="acceptBatch('${escapeHtml(item.batch_id)}')" class="success">确认 ${escapeHtml(taskType)}</button>
          <button onclick="rejectBatch('${escapeHtml(item.batch_id)}')" class="danger">reject</button>
          <button onclick="showManifest('${escapeHtml(item.batch_id)}')" class="secondary">manifest</button>
          <button onclick="showModal('batch 详情', state.batches.find(x => x.batch_id === '${escapeHtml(item.batch_id)}'))" class="secondary">详情</button>
        </div>
      </div>
    </div>`).join('');
  document.querySelectorAll('.batch-check').forEach(el => {
    el.onchange = (event) => {
      const id = event.target.dataset.id;
      if (event.target.checked) state.selectedBatchIds.add(id); else state.selectedBatchIds.delete(id);
    };
  });
}

function renderDatasets() {
  const root = $('datasets');
  if (!state.datasets.length) { root.className = 'list empty'; root.textContent = '暂无 dataset。'; return; }
  root.className = 'list';
  root.innerHTML = state.datasets.map(item => `
    <div class="item">
      <div class="item-title">${escapeHtml(item.dataset_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status)}</span></div>
      <div class="item-meta">${escapeHtml(item.task_type)} | batches=${(item.batch_ids || []).length} | images=${item.image_count || 0} | labels=${item.label_count || 0}</div>
      <div class="item-actions"><button onclick="showModal('dataset 详情（含源 manifest）', state.datasets.find(x => x.dataset_id === '${escapeHtml(item.dataset_id)}'))" class="secondary">详情</button></div>
    </div>`).join('');
}

function renderJobs() {
  const root = $('jobs');
  if (!state.jobs.length) { root.className = 'list empty'; root.textContent = '暂无 training job。'; return; }
  root.className = 'list';
  root.innerHTML = state.jobs.map(item => `
    <div class="item">
      <div class="item-title">${escapeHtml(item.job_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status)}</span></div>
      <div class="item-meta">stage=${escapeHtml(item.current_stage)} | dataset=${escapeHtml(item.dataset_id)} | model=${escapeHtml(item.output_model_package || '-')} | ${formatTime(item.updated_at_ms)}</div>
      <div class="item-actions">
        <button onclick="loadJobLogs('${escapeHtml(item.job_id)}', true)" class="secondary">日志</button>
        <button onclick="cancelJob('${escapeHtml(item.job_id)}')" class="warning">取消</button>
        <button onclick="showModal('job 详情', state.jobs.find(x => x.job_id === '${escapeHtml(item.job_id)}'))" class="secondary">详情</button>
      </div>
    </div>`).join('');
}

function renderModels() {
  const root = $('models');
  if (!state.models.length) { root.className = 'list empty'; root.textContent = '暂无模型包。训练任务成功后会自动生成。'; return; }
  root.className = 'list';
  root.innerHTML = state.models.map(item => `
    <div class="item">
      <div class="item-title">${escapeHtml(item.model_id)} <span class="badge ${badgeClass(item.status)}">${escapeHtml(item.status)}</span></div>
      <div class="item-meta">${escapeHtml(item.task_type)} | platform=${escapeHtml(item.target_platform)} | job=${escapeHtml(item.job_id || '-')}</div>
      <div class="item-actions">
        <button onclick="publishModel('${escapeHtml(item.model_id)}')" class="success">发布</button>
        <button onclick="showModal('model package 详情', state.models.find(x => x.model_id === '${escapeHtml(item.model_id)}'))" class="secondary">详情</button>
      </div>
    </div>`).join('');
}

function renderDevices() {
  const root = $('devices');
  if (!state.devices.length) { root.className = 'list empty'; root.textContent = '暂无设备。'; return; }
  root.className = 'list';
  root.innerHTML = state.devices.map(item => `
    <div class="item">
      <div class="item-title">${escapeHtml(item.device_id)} <span class="badge ${badgeClass(item.sync_status)}">${escapeHtml(item.sync_status || 'unknown')}</span></div>
      <div class="item-meta">${escapeHtml(item.device_type)} | ip=${escapeHtml(item.ip || '-')} | current=${escapeHtml(item.current_model || '-')} | target=${escapeHtml(item.target_model || '-')}</div>
      <div class="item-actions"><button onclick="showModal('device 详情', state.devices.find(x => x.device_id === '${escapeHtml(item.device_id)}'))" class="secondary">详情</button></div>
    </div>`).join('');
}

function refreshSelects() {
  const datasetSelect = $('datasetSelect');
  datasetSelect.innerHTML = state.datasets.length
    ? state.datasets.map(d => `<option value="${escapeHtml(d.dataset_id)}">${escapeHtml(d.dataset_id)} (${escapeHtml(d.task_type)})</option>`).join('')
    : '<option value="">请先构建 dataset</option>';
  datasetSelect.disabled = !state.datasets.length;

  $('assignDeviceSelect').innerHTML = state.devices.length
    ? state.devices.map(d => `<option value="${escapeHtml(d.device_id)}">${escapeHtml(d.device_id)}</option>`).join('')
    : '<option value="">请先登记设备</option>';
  $('assignModelSelect').innerHTML = state.models.length
    ? state.models.map(m => `<option value="${escapeHtml(m.model_id)}">${escapeHtml(m.model_id)}</option>`).join('')
    : '<option value="">请先生成模型包</option>';
}

function selectedTaskType() { return $('datasetTaskType').value; }

function showManifest(batchId) {
  const batch = state.batches.find(x => x.batch_id === batchId);
  if (!batch) return showToast('batch 不存在');
  showModal(`manifest：${batchId}`, batch.manifest || {message: '该 batch 未包含 manifest.json'});
}

async function processSelectedPackages() {
  const packages = [...state.selectedPackageNames];
  if (!packages.length) return showToast('请先勾选 incoming 目录下的 tar.gz 上传包。');
  const result = await api('/api/server/batches/process-incoming', {
    method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({packages})
  });
  state.selectedPackageNames.clear();
  showModal('上传包处理完成', result.batch);
  await refresh();
}
async function acceptBatch(batchId) {
  await api(`/api/server/batches/${encodeURIComponent(batchId)}/accept`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task_type: selectedTaskType()})
  });
  await refresh();
}
async function rejectBatch(batchId) {
  await api(`/api/server/batches/${encodeURIComponent(batchId)}/reject`, {method:'POST', body:'{}'});
  state.selectedBatchIds.delete(batchId);
  await refresh();
}
async function acceptSelected() {
  const ids = [...state.selectedBatchIds];
  if (!ids.length) return showToast('请先勾选 batch。');
  for (const id of ids) await api(`/api/server/batches/${encodeURIComponent(id)}/accept`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({task_type: selectedTaskType()})
  });
  await refresh();
}
async function rejectSelected() {
  const ids = [...state.selectedBatchIds];
  if (!ids.length) return showToast('请先勾选 batch。');
  for (const id of ids) await api(`/api/server/batches/${encodeURIComponent(id)}/reject`, {method:'POST', body:'{}'});
  state.selectedBatchIds.clear();
  await refresh();
}
async function buildDataset() {
  const taskType = selectedTaskType();
  const selected = [...state.selectedBatchIds].filter(id => {
    const item = state.batches.find(x => x.batch_id === id);
    return item && ['extracted', 'accepted'].includes(item.status);
  });
  const body = {task_type: taskType};
  if (selected.length) body.batch_ids = selected;
  const result = await api('/api/server/datasets/build', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  showModal('数据集已构建', result.dataset);
  await refresh();
}
async function loadJobLogs(jobId, show) {
  const result = await api(`/api/server/training/jobs/${encodeURIComponent(jobId)}/logs`);
  $('jobLog').textContent = result.logs || '暂无日志。';
  if (show) showModal(`任务日志：${jobId}`, result.logs || '暂无日志。');
}
async function cancelJob(jobId) {
  await api(`/api/server/training/jobs/${encodeURIComponent(jobId)}/cancel`, {method:'POST', body:'{}'});
  await refresh();
}
async function publishModel(modelId) {
  const publishRoot = $('publishRootInput').value.trim();
  const body = publishRoot ? {publish_root: publishRoot} : {};
  const result = await api(`/api/server/model-packages/${encodeURIComponent(modelId)}/publish`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  showModal('模型包已发布', result.publish);
  await refresh();
}

$('refreshBtn').onclick = () => refresh().catch(err => showToast(err.message));
$('modalClose').onclick = () => { $('modal').style.display = 'none'; };
$('modal').onclick = (event) => { if (event.target.id === 'modal') $('modal').style.display = 'none'; };
$('refreshIncomingBtn').onclick = () => refresh().catch(err => showToast(err.message));
$('processIncomingBtn').onclick = () => processSelectedPackages().catch(err => showToast(err.message));
$('acceptSelectedBtn').onclick = () => acceptSelected().catch(err => showToast(err.message));
$('rejectSelectedBtn').onclick = () => rejectSelected().catch(err => showToast(err.message));
$('buildDatasetBtn').onclick = () => buildDataset().catch(err => showToast(err.message));
$('datasetTaskType').onchange = () => renderBatches();
$('openAnnotatorBtn').onclick = () => showModal('标注器入口说明', `v3 服务端当前先按 v2 的数据流转方式完成 incoming 包处理和 batch/dataset 管理，内置标注器尚未迁移。\n\n当前建议流程：\n1. 边缘端 Web 打包 tar.gz 后，把文件复制或同步到服务端 incoming_root。\n2. 在第 1 步勾选 tar.gz 并处理，服务端会解压为 batch。\n3. 在第 2 步查看 batch manifest，根据实际数据类型选择 detection/classification/OBB/segmentation。\n4. 标注人员完成标注审核后，再确认 batch 并构建 dataset。\n\n后续接入 v2 标注器时，应继续复用 v3 的 batch/dataset API，不恢复 v2 的目录强绑定逻辑。`);

$('jobForm').onsubmit = async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const body = Object.fromEntries(form.entries());
  if (!body.dataset_id) return showToast('请先构建 dataset。');
  body.epochs = Number(body.epochs); body.batch_size = Number(body.batch_size); body.imgsz = Number(body.imgsz);
  const result = await api('/api/server/training/jobs', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  showModal('训练任务已创建', result.job);
  setTimeout(() => refresh().catch(console.error), 200);
  setTimeout(() => refresh().catch(console.error), 800);
};
$('deviceForm').onsubmit = async (event) => {
  event.preventDefault();
  const body = Object.fromEntries(new FormData(event.target).entries());
  const result = await api('/api/server/devices', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  showModal('设备已更新', result.device);
  await refresh();
};
$('assignModelBtn').onclick = async () => {
  const deviceId = $('assignDeviceSelect').value;
  const modelId = $('assignModelSelect').value;
  if (!deviceId || !modelId) return showToast('请先登记设备并生成模型包。');
  const result = await api(`/api/server/devices/${encodeURIComponent(deviceId)}/assign-model`, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({model_id: modelId})});
  showModal('目标模型已分配', result.device);
  await refresh();
};

refresh().catch(err => {
  $('serverBadge').textContent = '服务端：连接失败';
  $('serverBadge').className = 'badge danger';
  $('currentContext').textContent = err.message;
});
