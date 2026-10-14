<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🤖</span> 规则自动标注</h2>
      <p>基于内置规则与自定义规则对数据集版本真实执行自动标注，产物写入新版本（输入版本 → 输出版本）</p>
    </div>

    <div class="filter-bar" style="display: block">
      <el-form label-width="90px">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0 20px">
          <el-form-item label="任务名称"><el-input v-model="form.name" placeholder="如：静音分割标注" /></el-form-item>
          <el-form-item label="数据集" required>
            <el-select v-model="form.dataset_id" filterable style="width: 100%" @change="onDatasetChange">
              <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`" :value="d.id" />
            </el-select>
          </el-form-item>
          <el-form-item label="输入版本">
            <VersionSelect v-model="form.input_version_id" :dataset-id="form.dataset_id" />
          </el-form-item>
        </div>
        <el-form-item label="标注规则" required>
          <el-radio-group v-model="form.algorithm">
            <el-radio-button v-for="r in rules" :key="r.key" :value="r.key">{{ r.name }}</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item v-if="currentRule" label="规则参数">
          <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: center">
            <span class="muted">{{ currentRule.desc }}</span>
            <template v-for="(val, key) in currentRule.params" :key="key">
              <span style="font-size: 0.82rem">{{ key }}:</span>
              <el-input-number v-if="typeof val === 'number'" v-model="form.rules_config[key]" size="small" :step="Math.abs(val) >= 10 ? 1 : 0.1" style="width: 120px" />
              <el-input v-else-if="typeof val === 'string'" v-model="form.rules_config[key]" size="small" style="width: 140px" />
              <el-select v-else-if="Array.isArray(val)" v-model="form.rules_config[key]" multiple filterable allow-create default-first-option size="small" style="width: 220px" />
            </template>
            <span class="muted">适用：{{ currentRule.applicable?.join('/') }}</span>
            <el-tag v-if="currentRule.model_ready === false" type="danger" size="small">
              ⚠️ {{ currentRule.model_name }} 模型未就绪
            </el-tag>
            <el-tag v-else-if="currentRule.model_name" type="success" size="small">
              ✅ {{ currentRule.model_name }} 就绪
            </el-tag>
          </div>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :disabled="!auth.canOperate || currentRule?.model_ready === false" :loading="submitting" @click="onSubmit">
            🚀 创建并执行标注任务
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="table-card">
      <TaskTable ref="table" url="/auto-label" :version-map="versionMap">
        <el-table-column prop="name" label="任务名称" min-width="150" show-overflow-tooltip />
        <el-table-column label="类型" width="70"><template #default="{ row }">{{ TYPE_NAMES[row.data_type] }}</template></el-table-column>
        <el-table-column prop="algorithm" label="算法" width="130" />
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
const { datasets, versionMap, loadDatasets, currentVersionId, routeInput } = useDatasets()
const table = ref(null)
const submitting = ref(false)
const allRules = ref({})
const form = reactive({ name: '', dataset_id: null, input_version_id: null, algorithm: '', rules_config: {} })

const currentDataset = computed(() => datasets.value.find((d) => d.id === form.dataset_id))
const rules = computed(() => (currentDataset.value ? allRules.value[currentDataset.value.data_type] || [] : []))
const currentRule = computed(() => rules.value.find((r) => r.key === form.algorithm))

function onDatasetChange() {
  form.input_version_id = currentVersionId(form.dataset_id)  // 默认选中当前版本
  form.algorithm = rules.value[0]?.key || ''
  onRuleChange()
}
function onRuleChange() {
  const src = currentRule.value?.params || {}
  form.rules_config = Object.fromEntries(
    Object.entries(src).map(([k, v]) => [k, Array.isArray(v) ? [...v] : v]))
}
import { watch } from 'vue'
watch(() => form.algorithm, onRuleChange)

async function onSubmit() {
  if (!form.dataset_id) return ElMessage.error('请选择数据集')
  if (!form.algorithm) return ElMessage.error('请选择标注规则')
  submitting.value = true
  try {
    await api.post('/auto-label', {
      name: form.name || `${currentRule.value.name}任务`,
      data_type: currentDataset.value.data_type,
      dataset_id: form.dataset_id,
      algorithm: form.algorithm,
      rules_config: form.rules_config,
      input_version_id: form.input_version_id || undefined,
    })
    ElMessage.success('标注任务已创建，正在执行')
    table.value.reload()
  } catch { /* 已提示 */ } finally { submitting.value = false }
}

onMounted(async () => {
  allRules.value = await api.get('/auto-label/rules')
  await loadDatasets()
  const input=routeInput()
  if(input){form.dataset_id=input.dataset_id;onDatasetChange();form.input_version_id=input.version_id}
})
</script>

<style scoped>
.muted { color: #64748b; font-size: 0.82rem; }
</style>
