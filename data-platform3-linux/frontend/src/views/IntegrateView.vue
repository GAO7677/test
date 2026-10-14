<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🔗</span> 数据集成</h2>
      <p>支持跨数据集、跨模态集成（如图像+雷达→综合数据集）；产出数据集密级不得低于任一来源的最高密级</p>
    </div>

    <div class="filter-bar" style="display: block">
      <el-form label-width="90px">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0 20px">
          <el-form-item label="任务名称"><el-input v-model="form.name" placeholder="如：多来源文件合并" /></el-form-item>
          <el-form-item label="产出数据集" required>
            <el-select v-model="form.dataset_id" filterable style="width: 100%" @change="onDatasetChange">
              <el-option v-for="d in datasets" :key="d.id" :value="d.id"
                         :label="`${d.name}（${TYPE_NAMES[d.data_type]} · ${d.secret_level || '内部'}）`" />
            </el-select>
            <div class="muted" style="font-size: 0.78rem; margin-top: 4px">
              跨模态集成请先创建「综合」类型数据集；主输入版本须属于产出数据集
            </div>
          </el-form-item>
          <el-form-item label="主输入版本" required>
            <VersionSelect v-model="form.main_input_version_id" :dataset-id="form.dataset_id" />
          </el-form-item>
          <el-form-item label="其余输入版本">
            <el-select v-model="form.other_input_version_ids" multiple filterable style="width: 100%"
                       placeholder="可跨数据集、跨模态选择（按数据集分组）">
              <el-option-group v-for="d in datasets" :key="d.id"
                               :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`">
                <el-option v-for="v in versionsOf(d.id)" :key="v.id" :value="v.id"
                           :label="`${v.version}（${v.stage} · ${v.status === 'published' ? '已发布' : '草稿'}）`"
                           :disabled="v.id === form.main_input_version_id" />
              </el-option-group>
            </el-select>
          </el-form-item>
          <el-form-item label="融合类型">
            <el-radio-group v-model="form.combo_type">
              <el-radio-button value="raw">文件集合并</el-radio-button>
            </el-radio-group>
          </el-form-item>
          <el-form-item label="融合策略">
            <el-select v-model="form.strategy" style="width: 100%">
              <el-option label="文件集合并" value="raw" />
            </el-select>
          </el-form-item>
        </div>
        <el-form-item>
          <el-button type="primary" :disabled="!auth.canOperate" :loading="submitting" @click="onSubmit">🚀 创建并执行集成任务</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="table-card">
      <TaskTable ref="table" url="/integrate" :has-progress="false" :version-map="versionMap">
        <el-table-column prop="name" label="任务名称" min-width="140" show-overflow-tooltip />
        <el-table-column prop="combo_type" label="融合类型" width="100">
          <template #default="{ row }">{{ { raw: '文件集合并', feature: '特征级', decision: '决策级', semantic: '语义级' }[row.combo_type] || row.combo_type }}</template>
        </el-table-column>
        <el-table-column label="其余输入版本" min-width="140">
          <template #default="{ row }">
            <span v-if="row.detail?.other_input_version_ids?.length" class="muted">
              {{ row.detail.other_input_version_ids.map((id) => versionMap[id] || id).join('、') }}
            </span>
            <span v-else class="muted">—</span>
          </template>
        </el-table-column>
      </TaskTable>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { useDatasets } from '../composables'
import { TYPE_NAMES } from '../utils'
import TaskTable from '../components/TaskTable.vue'
import VersionSelect from '../components/VersionSelect.vue'

const auth = useAuthStore()
const { datasets, versionMap, datasetVersions, loadDatasets, currentVersionId, routeInput } = useDatasets()
const table = ref(null)
const submitting = ref(false)
const form = reactive({
  name: '', dataset_id: null, main_input_version_id: null,
  other_input_version_ids: [], combo_type: 'raw', strategy: 'raw',
})

function versionsOf(datasetId) {
  return [...(datasetVersions.value[datasetId] || [])].sort((a, b) => a.version_no - b.version_no)
}

function onDatasetChange() {
  form.main_input_version_id = currentVersionId(form.dataset_id)  // 默认当前版本
  form.other_input_version_ids = []
}

async function onSubmit() {
  if (!form.dataset_id) return ElMessage.error('请选择数据集')
  if (!form.main_input_version_id) return ElMessage.error('请指定主输入版本')
  submitting.value = true
  try {
    await api.post('/integrate', { name: form.name || '数据集成任务', ...form })
    ElMessage.success('集成任务已创建，正在执行')
    table.value.reload()
  } catch { /* 已提示 */ } finally { submitting.value = false }
}

onMounted(async()=>{
  await loadDatasets()
  const input=routeInput()
  if(input){form.dataset_id=input.dataset_id;await onDatasetChange();form.main_input_version_id=input.version_id}
})
</script>

<style scoped>
.muted { color: #64748b; }
</style>
