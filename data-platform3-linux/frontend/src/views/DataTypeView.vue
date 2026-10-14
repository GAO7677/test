<template>
  <div class="page">
    <div class="page-header">
      <h2><span>{{ meta.icon }}</span> {{ meta.title }}</h2>
      <p>{{ meta.desc }}</p>
    </div>

    <!-- 统计卡片 -->
    <div class="stat-cards">
      <div class="stat-card"><div class="stat-label">数据集总数</div><div class="stat-value">{{ stats.total ?? '-' }}</div></div>
      <div class="stat-card"><div class="stat-label">文件总数</div><div class="stat-value">{{ stats.total_files ?? '-' }}</div></div>
      <div class="stat-card"><div class="stat-label">总大小</div><div class="stat-value warning">{{ fmtSize(stats.total_size) }}</div></div>
      <div class="stat-card"><div class="stat-label">已标注</div><div class="stat-value success">{{ stats.by_status?.['已标注'] ?? 0 }}</div></div>
    </div>

    <!-- 筛选栏 -->
    <div class="filter-bar">
      <el-input v-model="query.search" placeholder="搜索名称…" clearable style="width: 180px" @change="load(1)" />
      <el-select v-model="query.project" placeholder="项目" clearable filterable style="width: 150px" @change="load(1)">
        <el-option v-for="p in projects" :key="p.id" :label="p.name" :value="p.name" />
      </el-select>
      <el-select v-model="query.scene" placeholder="场景" clearable filterable style="width: 130px" @change="load(1)">
        <el-option v-for="s in ['机载场景','地面场景','实验室','外场试验']" :key="s" :label="s" :value="s" />
      </el-select>
      <el-select v-model="query.source_type" placeholder="来源" clearable style="width: 110px" @change="load(1)">
        <el-option v-for="s in ['真实采集','仿真','AIGC']" :key="s" :label="s" :value="s" />
      </el-select>
      <el-select v-model="query.secret_level" placeholder="密级" clearable style="width: 110px" @change="load(1)">
        <el-option v-for="s in SECRET_LEVELS" :key="s" :label="s" :value="s" />
      </el-select>
      <el-select v-model="query.status" placeholder="状态" clearable style="width: 110px" @change="load(1)">
        <el-option v-for="s in ['未标注','待标注','已标注']" :key="s" :label="s" :value="s" />
      </el-select>
      <div style="margin-left: auto; display: flex; gap: 8px">
        <el-button v-if="auth.canWrite" type="primary" @click="openEdit()">➕ 新建{{ meta.short }}</el-button>
      </div>
    </div>

    <!-- 数据表格（可排序 + 分页） -->
    <div class="table-card" v-loading="loading">
      <el-table :data="rows" @sort-change="onSort" :default-sort="{ prop: 'created_at', order: 'descending' }">
        <el-table-column prop="name" label="名称" sortable="custom" min-width="180" show-overflow-tooltip />
        <el-table-column prop="format" label="格式" width="80" />
        <el-table-column v-for="col in meta.columns" :key="col.prop" :prop="col.prop" :label="col.label" :width="col.width">
          <template #default="{ row }">{{ col.fmt ? col.fmt(row[col.prop]) : (row[col.prop] ?? '-') }}</template>
        </el-table-column>
        <el-table-column prop="file_count" label="文件数" width="80" sortable="custom" />
        <el-table-column label="大小" width="100">
          <template #default="{ row }">{{ fmtSize(row.size_bytes) }}</template>
        </el-table-column>
        <el-table-column prop="project" label="项目" width="110" show-overflow-tooltip />
        <el-table-column prop="scene" label="场景" width="100" />
        <el-table-column prop="source_type" label="来源" width="90">
          <template #default="{ row }">
            <el-tag size="small" :type="row.source_type === 'AIGC' ? 'warning' : row.source_type === '仿真' ? 'info' : 'success'">{{ row.source_type || '真实采集' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="secret_level" label="密级" width="80">
          <template #default="{ row }">
            <el-tag size="small" effect="dark" :type="SECRET_TAGS[row.secret_level] || 'info'">{{ row.secret_level }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="90">
          <template #default="{ row }">
            <el-tag size="small" effect="dark" :type="{ 已标注: 'success', 待标注: 'warning', 未标注: 'info' }[row.status]">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="160" sortable="custom" />
        <el-table-column label="操作" width="390" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="openFiles(row)">文件</el-button>
            <el-button link type="primary" size="small" @click="openPreview(row)">预览</el-button>
            <el-button link type="primary" size="small" @click="$router.push(`/viz/${dataType}/${row.id}`)">可视化</el-button>
            <el-button v-if="auth.canWrite" link type="primary" size="small" @click="openUpload(row)">上传</el-button>
            <el-button v-if="auth.canWrite" link type="primary" size="small" :loading="annotatingId === row.id" :disabled="annotatingId !== null" @click="openOnlineLabel(row)">在线标注</el-button>
            <el-button v-if="auth.canWrite" link type="primary" size="small" @click="openLabel(row)">导入标注</el-button>
            <el-button link type="primary" size="small" @click="openDownload(row)">下载</el-button>
            <el-button v-if="auth.canWrite" link type="warning" size="small" @click="openEdit(row)">编辑</el-button>
            <el-button v-if="auth.canWrite" link type="danger" size="small" @click="onDelete(row)">删除</el-button>
          </template>
        </el-table-column>
        <template #empty><div class="empty-tip">暂无数据，点击右上角"新建{{ meta.short }}"开始</div></template>
      </el-table>
      <div class="pager">
        <el-pagination background layout="total, sizes, prev, pager, next" :total="total"
                       v-model:current-page="query.page" v-model:page-size="query.page_size"
                       :page-sizes="[10, 20, 50, 100]" @change="load()" />
      </div>
    </div>

    <!-- 新建/编辑 -->
    <el-dialog v-model="editVisible" :title="editRow ? '编辑' : '新建' + meta.short" width="520px">
      <el-form label-width="90px">
        <el-form-item label="名称" required><el-input v-model="form.name" /></el-form-item>
        <el-form-item label="格式">
          <el-select v-model="form.format" style="width: 100%">
            <el-option v-for="f in meta.formats" :key="f" :label="f" :value="f" />
          </el-select>
        </el-form-item>
        <el-form-item label="项目">
          <el-select v-model="form.project" style="width: 100%" allow-create filterable
                     placeholder="选择已有项目，或输入新项目名称">
            <el-option v-for="p in projects" :key="p.id" :label="p.name" :value="p.name" />
          </el-select>
        </el-form-item>
        <el-form-item label="场景">
          <el-select v-model="form.scene" style="width: 100%" allow-create filterable>
            <el-option v-for="s in ['机载场景','地面场景','实验室','外场试验']" :key="s" :label="s" :value="s" />
          </el-select>
        </el-form-item>
        <el-form-item label="数据来源">
          <el-select v-model="form.source_type" style="width: 100%" :disabled="!auth.canWrite">
            <el-option v-for="s in ['真实采集','仿真','AIGC']" :key="s" :label="s" :value="s" />
          </el-select>
        </el-form-item>
        <el-form-item label="密级">
          <el-radio-group v-model="form.secret_level">
            <el-radio-button v-for="s in SECRET_LEVELS" :key="s" :value="s">{{ s }}</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="标注类型"><el-input v-model="form.label_type" placeholder="如：目标检测 / 分割 / 分类" /></el-form-item>
        <el-form-item label="备注"><el-input v-model="form.remark" type="textarea" :rows="2" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="onSave">保存</el-button>
      </template>
    </el-dialog>

    <!-- 分片上传（文件/文件夹，串行队列 + 断点续传） -->
    <el-dialog v-model="uploadVisible" :title="`上传文件 — ${currentRow?.name || ''}`" width="620px">
      <el-alert type="info" :closable="false" style="margin-bottom: 12px"
                :title="`仅支持 ${meta.acceptDesc}；单文件与文件夹均可上传，目录结构自动还原；大文件自动 8MB 分片，支持断点续传；一次最多 5000 个文件 / 10GB${dataType === 'image' ? '；文件夹中的 instances*.json / annotations*.json（COCO 整表标注）将自动拆分导入' : ''}`" />
      <div style="display: flex; gap: 12px">
        <el-upload drag multiple :auto-upload="false" :show-file-list="false" :on-change="onPickFile" style="flex: 1">
          <el-icon size="40"><UploadFilled /></el-icon>
          <div class="el-upload__text">拖拽文件到此处，或 <em>点击选择文件</em></div>
        </el-upload>
        <div class="folder-pick" @click="folderInput.click()">
          <el-icon size="40"><FolderOpened /></el-icon>
          <div class="el-upload__text">选择<em>文件夹</em>上传<br /><span style="font-size: 0.72rem; color: var(--text-secondary)">保留子目录结构，串行稳定上传</span></div>
        </div>
        <input ref="folderInput" type="file" webkitdirectory multiple style="display: none" @change="onPickFolder" />
      </div>
      <div v-if="uploadTasks.length" style="margin-top: 10px; font-size: 0.8rem; color: var(--text-secondary)">
        队列：已完成 {{ uploadTasks.filter(t => t.status === 'done').length }} / {{ uploadTasks.length }}
        <el-button v-if="uploading" link type="danger" size="small" @click="uploadAbort = true">取消剩余</el-button>
      </div>
      <div v-for="t in uploadTasks" :key="t.key" style="margin-top: 8px">
        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px">
          <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 420px">{{ t.name }}（{{ fmtSize(t.size) }}）</span>
          <span>{{ t.status === 'done' ? '✅ 完成' : t.status === 'error' ? '❌ 失败' : t.status === 'waiting' ? '排队中' : t.progress + '%' }}</span>
        </div>
        <el-progress :percentage="t.progress" :status="t.status === 'error' ? 'exception' : t.status === 'done' ? 'success' : ''" />
      </div>
      <template #footer>
        <el-button @click="uploadVisible = false">关闭</el-button>
      </template>
    </el-dialog>

    <!-- 文件清单 -->
    <el-dialog v-model="filesVisible" :title="`文件清单 — ${currentRow?.name || ''}`" width="760px">
      <div style="display: flex; gap: 10px; margin-bottom: 10px">
        <el-input v-model="filesSearch" placeholder="按文件名搜索…" clearable size="small" style="width: 240px"
                  @change="loadFilesList(1)" @clear="loadFilesList(1)" />
      </div>
      <el-table :data="files" size="small" max-height="400" v-loading="filesLoading">
        <el-table-column prop="filename" label="文件名" min-width="240" show-overflow-tooltip />
        <el-table-column label="大小" width="110"><template #default="{ row }">{{ fmtSize(row.size_bytes) }}</template></el-table-column>
        <el-table-column prop="status" label="状态" width="90" />
        <el-table-column prop="created_at" label="上传时间" width="160" />
        <el-table-column label="操作" width="70">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="downloadOne(row)">下载</el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-empty v-if="!files.length && !filesLoading" description="暂无文件" :image-size="60" />
      <div style="margin-top: 10px; display: flex; justify-content: flex-end">
        <el-pagination background small layout="total, sizes, prev, pager, next" :total="filesTotal"
                       v-model:current-page="filesPage" v-model:page-size="filesPageSize"
                       :page-sizes="[20, 50, 100, 200]" @change="loadFilesList()" />
      </div>
    </el-dialog>

    <!-- 下载选择 -->
    <el-dialog v-model="downloadVisible" :title="`下载 — ${currentRow?.name || ''}`" width="440px">
      <el-form label-width="90px">
        <el-form-item label="下载内容">
          <el-radio-group v-model="downloadStage">
            <el-radio-button value="raw">原始数据</el-radio-button>
            <el-radio-button value="labeled">标注文件</el-radio-button>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="downloadVisible = false">取消</el-button>
        <el-button type="primary" @click="onDownload">打包下载 ZIP</el-button>
      </template>
    </el-dialog>

    <!-- 标注上传：单个 _label.json / 批量（多文件或文件夹）/ COCO 整表（仅图像） -->
    <el-dialog v-model="labelVisible" :title="`导入标注 — ${currentRow?.name || ''}`" width="600px">
      <el-radio-group v-model="labelMode" style="margin-bottom: 12px">
        <el-radio-button value="single">单个标注</el-radio-button>
        <el-radio-button value="batch">批量标注</el-radio-button>
        <el-radio-button v-if="dataType === 'image'" value="coco">COCO 整表</el-radio-button>
      </el-radio-group>
      <el-alert type="info" :closable="false" style="margin-bottom: 12px"
                :title="labelMode === 'coco'
                  ? '上传含全部图片标注的 COCO JSON，系统按 file_name 匹配自动拆分为每文件标注'
                  : labelMode === 'batch'
                    ? '一次选择多个 <数据文件名>_label.json（或整个标注文件夹），系统按文件名自动对应到数据集中的各文件；支持子目录'
                    : `标注文件统一为 _label.json 格式（与某个数据文件同名）${dataType === 'image' ? '' : '；COCO 整表仅适用于图像数据'}`" />

      <!-- 单个标注 -->
      <template v-if="labelMode === 'single'">
        <el-upload drag :auto-upload="false" :show-file-list="false" :on-change="onPickLabel" accept=".json">
          <el-icon size="36"><Document /></el-icon>
          <div class="el-upload__text">选择标注 JSON 文件</div>
        </el-upload>
        <div v-if="labelFile" style="margin-top: 10px; font-size: 0.85rem">已选择：{{ labelFile.name }}</div>
      </template>

      <!-- 批量标注 -->
      <template v-else-if="labelMode === 'batch'">
        <div style="display: flex; gap: 12px">
          <el-upload drag multiple :auto-upload="false" :show-file-list="false"
                     :on-change="onPickLabelBatch" accept=".json" style="flex: 1">
            <el-icon size="36"><Document /></el-icon>
            <div class="el-upload__text">选择<em>多个</em> _label.json</div>
          </el-upload>
          <div class="folder-pick" @click="labelFolderInput.click()">
            <el-icon size="36"><FolderOpened /></el-icon>
            <div class="el-upload__text">选择<em>标注文件夹</em><br />
              <span style="font-size: 0.72rem; color: var(--text-secondary)">保留子目录，按文件名自动对应</span></div>
          </div>
          <input ref="labelFolderInput" type="file" webkitdirectory multiple style="display: none"
                 @change="onPickLabelFolder" />
        </div>
        <div v-if="labelBatch.length" style="margin-top: 10px; font-size: 0.85rem">
          已选择 {{ labelBatch.length }} 个标注文件
          <el-button link type="danger" size="small" @click="labelBatch = []">清空</el-button>
        </div>
      </template>

      <!-- COCO 整表 -->
      <template v-else>
        <el-upload drag :auto-upload="false" :show-file-list="false" :on-change="onPickLabel" accept=".json">
          <el-icon size="36"><Document /></el-icon>
          <div class="el-upload__text">选择 COCO 整表 JSON</div>
        </el-upload>
        <div v-if="labelFile" style="margin-top: 10px; font-size: 0.85rem">已选择：{{ labelFile.name }}</div>
      </template>

      <template #footer>
        <el-button @click="labelVisible = false">取消</el-button>
        <el-button type="primary" :disabled="labelSubmitDisabled" :loading="saving" @click="onUploadLabel">
          {{ labelMode === 'batch' ? '批量导入' : '上传' }}
        </el-button>
      </template>
    </el-dialog>

    <!-- 预览（多文件切换 + 标注叠加） -->
    <el-dialog v-model="previewVisible" destroy-on-close :title="`数据预览 — ${preview?.file || ''}`" width="780px">
      <div v-if="previewLoading" v-loading="true" style="height: 200px" />
      <template v-else-if="preview">
        <div class="preview-bar">
          <el-button size="small" :disabled="previewIdx <= 0" @click="switchPreview(previewIdx - 1)">← 上一个</el-button>
          <el-select :model-value="preview.file_id" size="small" filterable style="width: 320px" @change="onPickPreviewFile">
            <el-option v-for="f in preview.files" :key="f.id" :label="f.name" :value="f.id" />
          </el-select>
          <el-button size="small" :disabled="previewIdx >= preview.files.length - 1" @click="switchPreview(previewIdx + 1)">下一个 →</el-button>
          <el-checkbox v-model="showLabel" style="margin-left: auto">显示标注</el-checkbox>
        </div>
        <div v-if="showLabel" style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 6px">
          标注来源：{{ preview.label ? (preview.label_source || '原始标注') : '该文件暂无标注' }}
        </div>
        <PreviewPane :type="dataType" :data="preview.data"
                     :video-src="dataType === 'video' ? `/api/video/${previewRow.id}/files/${preview.file_id}/download` : ''"
                     :label="showLabel ? preview.label : null" />
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { FolderOpened } from '@element-plus/icons-vue'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { fmtSize, TYPE_META, SECRET_LEVELS, SECRET_TAGS } from '../utils'
import PreviewPane from '../components/PreviewPane.vue'

const props = defineProps({ dataType: { type: String, required: true } })
const auth = useAuthStore()
const router = useRouter()
const meta = computed(() => TYPE_META[props.dataType])

const rows = ref([])
const total = ref(0)
const loading = ref(false)
const stats = ref({})
const projects = ref([])
const query = reactive({ page: 1, page_size: 20, search: '', project: '', scene: '', source_type: '', secret_level: '', status: '', sort: '' })

async function load(page) {
  if (page) query.page = page
  loading.value = true
  try {
    const data = await api.get(`/${props.dataType}`, { params: query })
    rows.value = data.items
    total.value = data.total
  } finally { loading.value = false }
}
async function loadStats() { stats.value = await api.get(`/${props.dataType}/stats`) }
async function loadProjects() { projects.value = (await api.get('/projects')).items }
function onSort({ prop, order }) {
  query.sort = order ? (order === 'descending' ? '-' : '') + prop : ''
  load()
}

// ---- 新建/编辑 ----
const editVisible = ref(false)
const editRow = ref(null)
const saving = ref(false)
const form = reactive({ name: '', format: '', project: '', scene: '', source_type: '真实采集', secret_level: '内部', status: '未标注', label_type: '', remark: '' })
function openEdit(row) {
  editRow.value = row || null
  Object.assign(form, row ? { ...row } : { name: '', format: meta.value.formats[0], project: '', scene: '', source_type: '真实采集', secret_level: '内部', status: '未标注', label_type: '', remark: '' })
  editVisible.value = true
}
async function onSave() {
  if (!form.name) return ElMessage.error('名称不能为空')
  saving.value = true
  try {
    // 输入了新项目名 → 先落库（幂等）
    if (form.project && !projects.value.some((p) => p.name === form.project)) {
      await api.post('/projects', { name: form.project })
    }
    if (editRow.value) await api.put(`/${props.dataType}/${editRow.value.id}`, form)
    else await api.post(`/${props.dataType}`, form)
    ElMessage.success('已保存')
    editVisible.value = false
    load(); loadStats(); loadProjects()
  } catch { /* 已提示 */ } finally { saving.value = false }
}
async function onDelete(row) {
  await ElMessageBox.confirm(`确认删除「${row.name}」？（软删除，文件保留）`, '删除确认', { type: 'warning' })
  await api.delete(`/${props.dataType}/${row.id}`)
  ElMessage.success('已删除')
  load(); loadStats()
}

// ---- 下载（选内容） ----
const downloadVisible = ref(false)
const downloadStage = ref('raw')
function openDownload(row) { currentRow.value = row; downloadStage.value = 'raw'; downloadVisible.value = true }
function onDownload() {
  window.open(`/api/${props.dataType}/${currentRow.value.id}/download?stage=${downloadStage.value}`, '_blank')
  downloadVisible.value = false
}
function downloadOne(f) {
  window.open(`/api/${props.dataType}/${currentRow.value.id}/files/${f.id}/download`, '_blank')
}

// ---- 分片上传（串行队列） ----
const uploadVisible = ref(false)
const currentRow = ref(null)
const uploadTasks = ref([])
const uploading = ref(false)
const uploadAbort = ref(false)
const folderInput = ref(null)
const CHUNK = 8 * 1024 * 1024
const MAX_FILES = 5000
const MAX_TOTAL = 10 * 1024 * 1024 * 1024
let taskSeq = 0

function openUpload(row) { currentRow.value = row; uploadTasks.value = []; uploadAbort.value = false; uploadVisible.value = true }

async function fileHash(file) {  // 断点续传标识：MD5
  const buf = await file.arrayBuffer()
  const hash = await crypto.subtle.digest('MD5', buf).catch(() => null)
  if (hash) return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, '0')).join('')
  return `${file.name}-${file.size}-${file.lastModified}`
}

function relDirOf(file) {
  // 文件夹上传：webkitRelativePath 形如 根目录/子目录/文件.jpg → 取目录部分
  const rel = file.webkitRelativePath || ''
  if (!rel.includes('/')) return ''
  return rel.split('/').slice(0, -1).join('/')
}

function enqueueFile(file) {
  if (uploadTasks.value.length >= MAX_FILES) {
    return ElMessage.error(`一次最多上传 ${MAX_FILES} 个文件，请分批`)
  }
  const totalSize = uploadTasks.value.reduce((s, t) => s + t.size, 0) + file.size
  if (totalSize > MAX_TOTAL) {
    return ElMessage.error('本次上传总大小超过 10GB，请分批')
  }
  uploadTasks.value.push(reactive({
    key: ++taskSeq, file, name: file.webkitRelativePath || file.name,
    relPath: relDirOf(file), size: file.size,
    progress: 0, status: 'waiting', hash: '',
  }))
  processQueue()
}

async function processQueue() {
  if (uploading.value) return
  uploading.value = true
  try {
    for (const task of uploadTasks.value) {
      if (uploadAbort.value) { if (task.status === 'waiting') task.status = 'error'; continue }
      if (task.status !== 'waiting') continue
      await uploadOne(task)
    }
  } finally { uploading.value = false }
}

async function uploadOne(task) {
  const file = task.file
  task.status = 'uploading'
  try {
    task.hash = await fileHash(file)
    const chunkTotal = Math.max(Math.ceil(file.size / CHUNK), 1)
    const init = await api.post(`/${props.dataType}/${currentRow.value.id}/upload/init`,
      { filename: file.name, size: file.size, chunk_total: chunkTotal, file_hash: task.hash, relative_path: task.relPath })
    const fileId = init.file_id
    let startChunk = init.chunk_received   // 断点续传：跳过已收分片
    for (let i = startChunk; i < chunkTotal; i++) {
      if (uploadAbort.value) throw new Error('已取消')
      const blob = file.slice(i * CHUNK, (i + 1) * CHUNK)
      const fd = new FormData()
      fd.append('file_id', fileId)
      fd.append('chunk_index', i)
      fd.append('chunk', blob)
      const r = await api.post(`/${props.dataType}/${currentRow.value.id}/upload/chunk`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' }, silent: true })
      task.progress = Math.round(r.chunk_received / r.chunk_total * 100)
    }
    await api.post(`/${props.dataType}/${currentRow.value.id}/upload/complete`, { file_id: fileId }, { silent: true })
    task.progress = 100
    task.status = 'done'
    load(); loadStats()
  } catch (e) {
    task.status = 'error'
    if (e?.friendlyMessage) ElMessage.error(`${task.name}：${e.friendlyMessage}`)
  }
}

function onPickFile(uploadFile) { enqueueFile(uploadFile.raw) }
function onPickFolder(e) {
  for (const f of e.target.files) enqueueFile(f)
  e.target.value = ''
}

// ---- 文件清单（分页 + 搜索） ----
const filesVisible = ref(false)
const files = ref([])
const filesLoading = ref(false)
const filesSearch = ref('')
const filesPage = ref(1)
const filesPageSize = ref(50)
const filesTotal = ref(0)
async function openFiles(row) {
  currentRow.value = row
  filesSearch.value = ''
  filesPage.value = 1
  filesVisible.value = true
  await loadFilesList(1)
}
async function loadFilesList(resetPage) {
  if (resetPage) filesPage.value = 1
  filesLoading.value = true
  try {
    const data = await api.get(`/${props.dataType}/${currentRow.value.id}/files`, {
      params: { search: filesSearch.value, page: filesPage.value, page_size: filesPageSize.value },
    })
    files.value = data.items
    filesTotal.value = data.total ?? data.items.length
  } finally { filesLoading.value = false }
}

// ---- 在线标注：原始数据与预处理版本使用各自的 ID ----
const annotatingId = ref(null)
async function openOnlineLabel(row) {
  annotatingId.value = row.id
  try {
    const endpoint = `/${props.dataType}/${row.id}/annotation-context`
    let context = await api.get(endpoint)
    if (!context.dataset_id) {
      await ElMessageBox.confirm(
        `首次在线标注将建立工作副本，复制 ${context.file_count} 个文件（${fmtSize(context.size_bytes)}），继承已有标注。原始文件保留；后续标注提交为新版本。`,
        `在线标注 — ${row.name}`,
        { confirmButtonText: '建立副本并标注', cancelButtonText: '取消', type: 'info' },
      )
      context = await api.post(endpoint, {}, { timeout: 0 })
    }
    await router.push({ path: '/manual-label', query: {
      dataset_id: context.dataset_id, version_id: context.version_id,
    } })
  } catch { /* 取消或接口错误已提示 */ }
  finally { annotatingId.value = null }
}

// ---- 标注上传 ----
const labelVisible = ref(false)
const labelFile = ref(null)
const labelBatch = ref([])
const labelFolderInput = ref(null)
const labelMode = ref('single')
const labelSubmitDisabled = computed(() =>
  labelMode.value === 'batch' ? !labelBatch.value.length : !labelFile.value)
function openLabel(row) {
  currentRow.value = row
  labelFile.value = null
  labelBatch.value = []
  // COCO 整表仅适用于图像数据，其余模态从单个标注开始
  labelMode.value = 'single'
  labelVisible.value = true
}
function onPickLabel(f) { labelFile.value = f.raw }
function onPickLabelBatch(f) { if (f?.raw) labelBatch.value.push(f.raw) }
function onPickLabelFolder(e) {
  for (const f of e.target.files) labelBatch.value.push(f)
  e.target.value = ''
}
async function onUploadLabel() {
  saving.value = true
  try {
    const fd = new FormData()
    if (labelMode.value === 'batch') {
      // 批量导入：一次提交多个 <数据文件名>_label.json（含文件夹相对路径），后端按文件名自动对应
      for (const f of labelBatch.value) {
        fd.append('files', f)
        fd.append('rel_paths', f.webkitRelativePath || '')
      }
      const res = await api.post(`/${props.dataType}/${currentRow.value.id}/labels/batch-import`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' } })
      ElMessage.success(`批量导入完成：成功匹配 ${res.matched} 个文件` +
        (res.unmatched_count ? `，${res.unmatched_count} 个未匹配` : '') +
        (res.error_count ? `，${res.error_count} 个格式错误` : ''))
      if (res.unmatched_count) {
        ElMessage.warning('未匹配到数据文件：' + res.unmatched.slice(0, 5).join('、'))
      }
    } else if (labelMode.value === 'coco' && props.dataType === 'image') {
      fd.append('file', labelFile.value)
      const res = await api.post(`/${props.dataType}/${currentRow.value.id}/labels/import`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' } })
      ElMessage.success(
        `COCO 标注已导入：匹配 ${res.matched_files} 个文件` +
        (res.unmatched_annotations ? `，${res.unmatched_annotations} 条标注未匹配到文件` : ''))
    } else {
      fd.append('label', labelFile.value)
      await api.post(`/${props.dataType}/${currentRow.value.id}/label`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' } })
      ElMessage.success('标注文件已上传')
    }
    labelVisible.value = false
    load(); loadStats()
  } catch { /* 已提示 */ } finally { saving.value = false }
}

// ---- 预览（多文件切换 + 标注叠加） ----
const previewVisible = ref(false)
const previewLoading = ref(false)
const preview = ref(null)
const previewIdx = ref(0)
const showLabel = ref(true)
const previewRow = ref(null)
async function fetchPreview(row, fileId) {
  previewLoading.value = true
  try {
    const params = fileId ? { file_id: fileId } : {}
    const data = await api.get(`/${props.dataType}/${row.id}/preview`, { params })
    preview.value = data
    previewIdx.value = Math.max(data.files.findIndex((f) => f.id === data.file_id), 0)
  } catch { previewVisible.value = false } finally { previewLoading.value = false }
}
async function openPreview(row) {
  previewVisible.value = true
  preview.value = null
  previewRow.value = row
  showLabel.value = true
  await fetchPreview(row)
}
function switchPreview(idx) {
  const f = preview.value?.files?.[idx]
  if (f) fetchPreview(previewRow.value, f.id)
}
function onPickPreviewFile(fileId) { fetchPreview(previewRow.value, fileId) }

onMounted(() => { load(); loadStats(); loadProjects() })
</script>

<style scoped>
.folder-pick {
  flex: 1; border: 1px dashed var(--border-color, #3a4a63); border-radius: 6px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  cursor: pointer; padding: 10px; color: var(--text-secondary);
}
.folder-pick:hover { border-color: var(--el-color-primary); color: var(--el-color-primary); }
.preview-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
</style>
