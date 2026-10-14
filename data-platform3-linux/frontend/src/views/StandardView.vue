<template>
  <div class="page">
    <div class="page-header">
      <h2><span>📋</span> 参考标准管理</h2>
      <p>内置国标/ISO 标准 + 自定义标准；支持上传标准原文（PDF/Word/图片，≤50MB），自定义标准按"指标+阈值"规则参与质量评估</p>
    </div>

    <div class="filter-bar">
      <el-input v-model="search" placeholder="搜索编号 / 名称…" clearable style="width: 220px" />
      <div style="margin-left: auto">
        <el-button v-if="auth.canWrite" type="primary" @click="openCreate">➕ 新增标准</el-button>
      </div>
    </div>

    <div class="table-card" v-loading="loading">
      <el-table :data="filteredRows">
        <el-table-column prop="code" label="编号" width="180" show-overflow-tooltip />
        <el-table-column prop="name" label="名称" min-width="220" show-overflow-tooltip />
        <el-table-column label="类型" width="110">
          <template #default="{ row }">
            <el-tag v-if="row.is_custom" size="small" effect="dark" type="warning">自定义</el-tag>
            <el-tag v-else size="small" effect="plain" :type="row.standard_type === '国际标准' ? 'warning' : 'primary'">{{ row.standard_type }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="评估方式" min-width="200">
          <template #default="{ row }">
            <template v-if="row.is_custom && row.rules?.length">
              <el-tag v-for="(r, i) in row.rules" :key="i" size="small" effect="plain" style="margin: 2px 4px 2px 0">
                {{ r.name || metricName(r.metric) }} {{ OPS_NAMES[r.op] || r.op }} {{ r.threshold }}
              </el-tag>
            </template>
            <span v-else class="muted">六维加权（内置）</span>
          </template>
        </el-table-column>
        <el-table-column label="标准原文" width="170">
          <template #default="{ row }">
            <template v-if="row.has_file">
              <el-button link type="primary" size="small" @click="onDownloadFile(row)">📄 下载原文</el-button>
            </template>
            <el-upload v-if="auth.canWrite" :show-file-list="false" :auto-upload="false"
                       accept=".pdf,.doc,.docx,.png,.jpg,.jpeg,.bmp,.tiff"
                       :on-change="(f) => onUploadFile(row, f)">
              <el-button link size="small" :type="row.has_file ? 'warning' : 'primary'">
                {{ row.has_file ? '重新上传' : '上传原文' }}
              </el-button>
            </el-upload>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="80" />
        <el-table-column label="操作" width="80">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="onDownload(row)">导出记录</el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 新增标准（含自定义规则编辑） -->
    <el-dialog v-model="createVisible" title="新增参考标准" width="640px">
      <el-form label-width="90px">
        <el-form-item label="编号" required><el-input v-model="form.code" placeholder="如：Q/XXX 001-2026" /></el-form-item>
        <el-form-item label="名称" required><el-input v-model="form.name" /></el-form-item>
        <el-form-item label="类型">
          <el-radio-group v-model="form.standard_type">
            <el-radio-button value="国家标准">国家标准</el-radio-button>
            <el-radio-button value="国际标准">国际标准</el-radio-button>
            <el-radio-button value="行业标准">行业标准</el-radio-button>
            <el-radio-button value="自定义">自定义</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="适用维度"><el-input v-model="form.dimensions" /></el-form-item>
        <el-form-item label="适用范围"><el-input v-model="form.scope" /></el-form-item>
        <el-form-item label="内容摘要"><el-input v-model="form.content" type="textarea" :rows="2" /></el-form-item>

        <el-form-item label="评估规则">
          <div style="width: 100%">
            <div class="muted" style="font-size: 0.78rem; margin-bottom: 6px">
              每条规则 = 质量指标 + 比较符 + 阈值；合规率 = 通过规则数 / 总规则数。留空则按内置六维加权评估。
            </div>
            <div v-for="(r, i) in form.rules" :key="i" class="rule-row">
              <el-select v-model="r.metric" filterable placeholder="指标" style="width: 190px">
                <el-option v-for="m in QUALITY_METRICS" :key="m.key" :label="m.name" :value="m.key" />
              </el-select>
              <el-select v-model="r.op" style="width: 80px">
                <el-option v-for="(n, k) in OPS_NAMES" :key="k" :label="n" :value="k" />
              </el-select>
              <el-input-number v-model="r.threshold" size="default" style="width: 110px" />
              <el-input v-model="r.name" placeholder="规则名（可选）" style="width: 150px" />
              <el-button link type="danger" size="small" @click="form.rules.splice(i, 1)">✕</el-button>
            </div>
            <el-button size="small" @click="form.rules.push({ metric: 'file_count', op: '>=', threshold: 1, name: '' })">
              ➕ 添加规则
            </el-button>
          </div>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="onSave">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { QUALITY_METRICS } from '../utils'

const OPS_NAMES = { '>=': '≥', '<=': '≤', '>': '＞', '<': '＜', '==': '＝' }

const auth = useAuthStore()
const rows = ref([])
const loading = ref(false)
const search = ref('')
const createVisible = ref(false)
const saving = ref(false)
const form = reactive({ code: '', name: '', standard_type: '国家标准', dimensions: '', scope: '', content: '', rules: [] })

const filteredRows = computed(() => {
  const kw = search.value.trim().toLowerCase()
  if (!kw) return rows.value
  return rows.value.filter((r) => (r.code + r.name).toLowerCase().includes(kw))
})

function metricName(key) {
  return QUALITY_METRICS.find((m) => m.key === key)?.name || key
}

async function load() {
  loading.value = true
  try {
    const data = await api.get('/standards')
    rows.value = data.items
  } finally { loading.value = false }
}

function openCreate() {
  form.rules = []
  createVisible.value = true
}

async function onSave() {
  if (!form.code || !form.name) return ElMessage.error('编号与名称不能为空')
  for (const r of form.rules) {
    if (!r.metric || r.threshold === null || r.threshold === undefined)
      return ElMessage.error('请补全每条规则的指标与阈值')
  }
  saving.value = true
  try {
    await api.post('/standards', {
      ...form,
      standard_type: form.standard_type === '自定义' ? '自定义标准' : form.standard_type,
      rules: form.rules,
    })
    ElMessage.success('标准已新增')
    createVisible.value = false
    load()
  } catch { /* 已提示 */ } finally { saving.value = false }
}

async function onUploadFile(row, uploadFile) {
  const raw = uploadFile.raw
  if (!raw) return
  if (raw.size > 50 * 1024 * 1024) return ElMessage.error('原文文件不能超过 50MB')
  const fd = new FormData()
  fd.append('file', raw, raw.name)
  try {
    await api.post(`/standards/${row.id}/file`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
    ElMessage.success(`「${row.code}」原文已上传`)
    load()
  } catch { /* 已提示 */ }
}

function onDownloadFile(row) {
  window.open(`/api/standards/${row.id}/file`, '_blank')
}

function onDownload(row) {
  window.open(`/api/standards/${row.id}/download`, '_blank')
}

onMounted(load)
</script>

<style scoped>
.muted { color: #64748b; }
.rule-row { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }
</style>
