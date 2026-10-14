<template>
  <div class="page">
    <div class="page-header">
      <h2><span>✂️</span> 数据采样</h2>
      <p>训练 / 验证 / 测试集三划分（比例和 = 100%），真实复制文件到输出版本目录的 train / val / test 子目录</p>
    </div>

    <div class="filter-bar" style="display: block">
      <el-form label-width="90px">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0 20px">
          <el-form-item label="任务名称"><el-input v-model="form.name" placeholder="如：语音训练集划分" /></el-form-item>
          <el-form-item label="数据集" required>
            <el-select v-model="form.dataset_id" filterable style="width: 100%" @change="form.input_version_id = currentVersionId(form.dataset_id)">
              <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`" :value="d.id" />
            </el-select>
          </el-form-item>
          <el-form-item label="输入版本">
            <VersionSelect v-model="form.input_version_id" :dataset-id="form.dataset_id" />
          </el-form-item>
          <el-form-item label="划分方式">
            <el-select v-model="form.method" style="width: 100%">
              <el-option label="随机划分" value="random" />
            </el-select>
          </el-form-item>
        </div>
        <el-form-item label="划分比例" required>
          <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap">
            <span>训练集</span>
            <el-input-number v-model="form.train_ratio" :min="0" :max="100" size="small" />
            <span>验证集</span>
            <el-input-number v-model="form.val_ratio" :min="0" :max="100" size="small" />
            <span>测试集</span>
            <el-input-number v-model="form.test_ratio" :min="0" :max="100" size="small" />
            <el-tag :type="ratioOk ? 'success' : 'danger'" effect="dark">
              合计 {{ form.train_ratio + form.val_ratio + form.test_ratio }}%{{ ratioOk ? ' ✓' : '（须=100%）' }}
            </el-tag>
          </div>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :disabled="!auth.canOperate || !ratioOk" :loading="submitting" @click="onSubmit">
            🚀 创建并执行采样任务
          </el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="table-card">
      <TaskTable ref="table" url="/sample" :has-progress="false" :version-map="versionMap">
        <el-table-column prop="name" label="任务名称" min-width="140" show-overflow-tooltip />
        <el-table-column label="类型" width="70"><template #default="{ row }">{{ TYPE_NAMES[row.data_type] }}</template></el-table-column>
        <el-table-column label="比例" width="100">
          <template #default="{ row }">{{ row.train_ratio }}:{{ row.val_ratio }}:{{ row.test_ratio }}</template>
        </el-table-column>
        <el-table-column label="划分结果" min-width="180">
          <template #default="{ row }">
            <template v-if="row.status === 'completed'">
              <span style="font-size: 0.8rem">
                训练 {{ row.train_count }} · 验证 {{ row.val_count }} · 测试 {{ row.test_count }}
              </span>
              <div v-if="row.error_msg" style="font-size: 0.75rem; color: #34d399">{{ row.error_msg }}</div>
            </template>
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
const { datasets, versionMap, loadDatasets, currentVersionId, routeInput } = useDatasets()
const table = ref(null)
const submitting = ref(false)
const form = reactive({
  name: '', dataset_id: null, input_version_id: null, method: 'random',
  train_ratio: 70, val_ratio: 10, test_ratio: 20,
})
// 前端比例和=100%校验（后端同步校验）
const ratioOk = computed(() => form.train_ratio + form.val_ratio + form.test_ratio === 100)

async function onSubmit() {
  if (!form.dataset_id) return ElMessage.error('请选择数据集')
  if (!ratioOk.value) return ElMessage.error('训练/验证/测试比例之和须等于 100%')
  const ds = datasets.value.find((d) => d.id === form.dataset_id)
  submitting.value = true
  try {
    await api.post('/sample', {
      name: form.name || '数据采样任务', data_type: ds.data_type, ...form,
      input_version_id: form.input_version_id || undefined,
    })
    ElMessage.success('采样任务已创建，正在执行')
    table.value.reload()
  } catch { /* 已提示 */ } finally { submitting.value = false }
}

onMounted(async()=>{
  await loadDatasets()
  const input=routeInput()
  if(input){form.dataset_id=input.dataset_id;form.input_version_id=input.version_id}
})
</script>
