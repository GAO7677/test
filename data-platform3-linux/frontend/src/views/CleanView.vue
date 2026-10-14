<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🧹</span> 数据清洗</h2>
      <p>缺失值处理、异常值检测、重复清理、格式标准化——读取版本文件真实执行，生成清洗后新版本</p>
    </div>

    <div class="filter-bar" style="display: block">
      <el-form label-width="90px">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0 20px">
          <el-form-item label="任务名称"><el-input v-model="form.name" placeholder="如：语音数据清洗" /></el-form-item>
          <el-form-item label="数据集" required>
            <el-select v-model="form.dataset_id" filterable style="width: 100%" @change="form.input_version_id = currentVersionId(form.dataset_id)">
              <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`" :value="d.id" />
            </el-select>
          </el-form-item>
          <el-form-item label="输入版本">
            <VersionSelect v-model="form.input_version_id" :dataset-id="form.dataset_id" />
          </el-form-item>
        </div>
        <el-form-item label="清洗规则">
          <el-checkbox v-model="form.rules.missing">缺失值处理（去除空行/空文件）</el-checkbox>
          <el-checkbox v-model="form.rules.anomaly">异常值检测（IQR/损坏文件剔除）</el-checkbox>
          <el-checkbox v-model="form.rules.duplicate">重复清理（内容哈希去重）</el-checkbox>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :disabled="!auth.canOperate" :loading="submitting" @click="onSubmit">🚀 创建并执行清洗任务</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="table-card">
      <TaskTable ref="table" url="/clean" :version-map="versionMap">
        <el-table-column prop="name" label="任务名称" min-width="140" show-overflow-tooltip />
        <el-table-column label="类型" width="70"><template #default="{ row }">{{ TYPE_NAMES[row.data_type] }}</template></el-table-column>
        <el-table-column label="清洗结果" min-width="220">
          <template #default="{ row }">
            <span v-if="row.stats_after && row.status === 'completed'" style="font-size: 0.78rem">
              去重 {{ row.stats_after.removed_duplicates || 0 }} · 异常 {{ row.stats_after.removed_anomalies || 0 }} · 缺失 {{ row.stats_after.removed_missing || 0 }}
              （{{ row.stats_before?.files ?? '?' }} → {{ row.stats_after.files }} 文件）
            </span>
          </template>
        </el-table-column>
      </TaskTable>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { useDatasets } from '../composables'
import { TYPE_NAMES } from '../utils'
import TaskTable from '../components/TaskTable.vue'
import VersionSelect from '../components/VersionSelect.vue'

const auth = useAuthStore()
const { datasets, versionMap, loadDatasets, currentVersionId, routeInput } = useDatasets()
const table = ref(null)
const submitting = ref(false)
const form = reactive({
  name: '', dataset_id: null, input_version_id: null,
  rules: { missing: true, anomaly: true, duplicate: true },
})

async function onSubmit() {
  if (!form.dataset_id) return ElMessage.error('请选择数据集')
  const ds = datasets.value.find((d) => d.id === form.dataset_id)
  submitting.value = true
  try {
    await api.post('/clean', {
      name: form.name || '数据清洗任务', data_type: ds.data_type,
      dataset_id: form.dataset_id, rules: form.rules,
      input_version_id: form.input_version_id || undefined,
    })
    ElMessage.success('清洗任务已创建，正在执行')
    table.value.reload()
  } catch { /* 已提示 */ } finally { submitting.value = false }
}

onMounted(async()=>{
  await loadDatasets()
  const input=routeInput()
  if(input){form.dataset_id=input.dataset_id;form.input_version_id=input.version_id}
})
</script>
