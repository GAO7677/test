<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🔄</span> 数据转换</h2>
      <p>图像和音频格式转换、图像尺寸调整、音频峰值归一化、CSV与PLY点云转换；不支持的组合提交前提示</p>
    </div>

    <div class="filter-bar" style="display: block">
      <el-form label-width="90px">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0 20px">
          <el-form-item label="任务名称"><el-input v-model="form.name" /></el-form-item>
          <el-form-item label="数据集" required>
            <el-select v-model="form.dataset_id" filterable style="width: 100%" @change="form.input_version_id = currentVersionId(form.dataset_id)">
              <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`" :value="d.id" />
            </el-select>
          </el-form-item>
          <el-form-item label="输入版本">
            <VersionSelect v-model="form.input_version_id" :dataset-id="form.dataset_id" />
          </el-form-item>
          <el-form-item label="转换类型" required>
            <el-radio-group v-model="form.transform_type">
              <el-radio-button value="format">格式转换</el-radio-button>
              <el-radio-button value="standardize">标准化</el-radio-button>
              
            </el-radio-group>
          </el-form-item>
          <el-form-item v-if="form.transform_type === 'format'" label="目标格式">
            <el-select v-model="form.target_format" style="width: 100%">
              <el-option v-for="f in targetFormats" :key="f" :label="f.toUpperCase()" :value="f" />
            </el-select>
          </el-form-item>
          <el-form-item v-if="form.transform_type === 'standardize'" label="处理参数">
            <el-input-number v-if="currentDataset?.data_type === 'image'" v-model="form.params.size" :min="1" :max="8192" />
            <span v-if="currentDataset?.data_type === 'image'">最长边像素，保留比例</span>
            <span v-else>音频峰值归一化至 0.89</span>
          </el-form-item>
        </div>
        <el-form-item>
          <el-button type="primary" :disabled="!auth.canOperate" :loading="submitting" @click="onSubmit">🚀 创建并执行转换任务</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="table-card">
      <TaskTable ref="table" url="/transform" :version-map="versionMap">
        <el-table-column prop="name" label="任务名称" min-width="140" show-overflow-tooltip />
        <el-table-column label="类型" width="70"><template #default="{ row }">{{ TYPE_NAMES[row.data_type] }}</template></el-table-column>
        <el-table-column prop="transform_type" label="转换类型" width="100">
          <template #default="{ row }">{{ { format: '格式转换', standardize: '标准化', feature: '特征提取', pipeline: '流水线' }[row.transform_type] }}</template>
        </el-table-column>
        <el-table-column prop="target_format" label="目标格式" width="90">
          <template #default="{ row }">{{ row.target_format?.toUpperCase() || '—' }}</template>
        </el-table-column>
      </TaskTable>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
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
  transform_type: 'format', target_format: 'mp3', params: { size: 640, target_peak: 0.89 },
})

const currentDataset = computed(() => datasets.value.find((d) => d.id === form.dataset_id))
const targetFormats = computed(() => {
  const t = currentDataset.value?.data_type
  return { voice: ['wav', 'mp3', 'flac'], radar: ['csv', 'ply'],
           image: ['jpg', 'png'], video: [], text: [] }[t] || ['wav', 'mp3']
})

watch(() => currentDataset.value?.data_type, () => {
  if (!targetFormats.value.includes(form.target_format)) form.target_format = targetFormats.value[0] || ''
})

async function onSubmit() {
  if (!form.dataset_id) return ElMessage.error('请选择数据集')
  submitting.value = true
  try {
    await api.post('/transform', {
      name: form.name || '数据转换任务', data_type: currentDataset.value.data_type,
      dataset_id: form.dataset_id, transform_type: form.transform_type,
      target_format: targetFormats.value.includes(form.target_format) ? form.target_format : (targetFormats.value[0] || ""), params: form.params,
      input_version_id: form.input_version_id || undefined,
    })
    ElMessage.success('转换任务已创建，正在执行')
    table.value.reload()
  } catch { /* 已提示 */ } finally { submitting.value = false }
}

onMounted(async()=>{
  await loadDatasets()
  const input=routeInput()
  if(input){form.dataset_id=input.dataset_id;form.input_version_id=input.version_id}
})
</script>
