<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🧩</span> 工作流</h2>
      <p>预处理任务节点的顺序执行链：上游输出版本自动作为下游输入版本，任一节点失败即中止</p>
    </div>

    <div class="filter-bar" style="display: block">
      <el-form label-width="90px">
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0 20px">
          <el-form-item label="工作流名称"><el-input v-model="form.name" placeholder="如：标注-清洗-采样流水线" /></el-form-item>
          <el-form-item label="数据集" required>
            <el-select v-model="form.dataset_id" filterable style="width: 100%" @change="onDatasetChange">
              <el-option v-for="d in datasets" :key="d.id" :label="`${d.name}（${TYPE_NAMES[d.data_type]}）`" :value="d.id" />
            </el-select>
          </el-form-item>
          <el-form-item label="起始版本">
            <VersionSelect v-model="form.input_version_id" :dataset-id="form.dataset_id" />
          </el-form-item>
        </div>

        <!-- 节点编排（可上下移动排序） -->
        <el-form-item label="执行节点">
          <div style="width: 100%">
            <div v-for="(node, i) in form.nodes" :key="i" class="node-row">
              <el-tag effect="dark" type="primary">{{ i + 1 }}</el-tag>
              <el-select v-model="node.type" style="width: 140px">
                <el-option label="自动标注" value="auto_label" />
                <el-option label="数据清洗" value="clean" />
                <el-option label="数据转换" value="transform" />
                <el-option label="数据采样" value="sample" />
              </el-select>
              <el-select v-if="node.type === 'auto_label'" v-model="node.algorithm" placeholder="算法" style="width: 150px">
                <el-option v-for="r in labelRules" :key="r.key" :label="r.model_ready === false ? r.name + '（未就绪）' : r.name" :value="r.key" :disabled="r.model_ready === false" />
              </el-select>
              <template v-if="node.type === 'transform'">
                <el-select v-model="node.transform_type" style="width: 110px">
                  <el-option label="格式转换" value="format" /><el-option label="标准化" value="standardize" />
                </el-select>
                <el-input v-model="node.target_format" placeholder="目标格式" style="width: 100px" />
              </template>
              <template v-if="node.type === 'sample'">
                <el-input-number v-model="node.params.train_ratio" size="small" :min="0" :max="100" style="width: 80px" />
                <el-input-number v-model="node.params.val_ratio" size="small" :min="0" :max="100" style="width: 80px" />
                <el-input-number v-model="node.params.test_ratio" size="small" :min="0" :max="100" style="width: 80px" />
              </template>
              <el-button link size="small" :disabled="i === 0" @click="move(i, -1)">↑</el-button>
              <el-button link size="small" :disabled="i === form.nodes.length - 1" @click="move(i, 1)">↓</el-button>
              <el-button link type="danger" size="small" @click="form.nodes.splice(i, 1)">✕</el-button>
            </div>
            <el-button size="small" @click="addNode">➕ 添加节点</el-button>
          </div>
        </el-form-item>
        <el-form-item>
          <el-button :disabled="!auth.canOperate" @click="onSave(false)">💾 保存工作流</el-button>
          <el-button type="primary" :disabled="!auth.canOperate" :loading="submitting" @click="onSave(true)">🚀 保存并执行</el-button>
        </el-form-item>
      </el-form>
    </div>

    <div class="table-card">
      <TaskTable ref="table" url="/workflows" :has-progress="false" :version-map="versionMap">
        <el-table-column prop="name" label="工作流名称" min-width="150" show-overflow-tooltip />
        <el-table-column label="节点" min-width="200">
          <template #default="{ row }">
            <el-tag v-for="(n, i) in row.nodes || []" :key="i" size="small" effect="plain" style="margin-right: 6px">
              {{ { auto_label: '自动标注', clean: '清洗', transform: '转换', sample: '采样' }[n.type] }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="100">
          <template #default="{ row }">
            <el-button v-if="auth.canOperate && row.status !== 'running'" link type="primary" size="small" @click="onRun(row)">执行</el-button>
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
const labelRules = ref([])
const form = reactive({ name: '', dataset_id: null, input_version_id: null, nodes: [] })

const currentDataset = computed(() => datasets.value.find((d) => d.id === form.dataset_id))

function addNode() {
  form.nodes.push({ type: 'clean', algorithm: '', transform_type: 'format', target_format: '',
                    params: { train_ratio: 70, val_ratio: 10, test_ratio: 20 } })
}
function move(i, dir) {
  const [n] = form.nodes.splice(i, 1)
  form.nodes.splice(i + dir, 0, n)
}

async function onSave(run) {
  if (!form.dataset_id) return ElMessage.error('请选择数据集')
  if (!form.nodes.length) return ElMessage.error('请至少添加一个执行节点')
  for (const n of form.nodes) {
    if (n.type === 'sample' && n.params.train_ratio + n.params.val_ratio + n.params.test_ratio !== 100)
      return ElMessage.error('采样节点的三比例之和须等于 100%')
  }
  submitting.value = true
  try {
    const wf = await api.post('/workflows', {
      name: form.name || '未命名工作流', data_type: currentDataset.value.data_type,
      dataset_id: form.dataset_id, nodes: form.nodes, edges: [],
      input_version_id: form.input_version_id || undefined,
    })
    if (run) {
      await api.post(`/workflows/${wf.id}/run`)
      ElMessage.success('工作流已启动')
    } else {
      ElMessage.success('工作流已保存')
    }
    table.value.reload()
  } catch { /* 已提示 */ } finally { submitting.value = false }
}

async function onRun(row) {
  await api.post(`/workflows/${row.id}/run`)
  ElMessage.success('工作流已启动')
  table.value.reload()
}

// 按所选数据集的模态加载可用标注算法（不再固定语音规则）
async function onDatasetChange() {
  form.input_version_id = currentVersionId(form.dataset_id)
  const dtype = currentDataset.value?.data_type
  if (dtype) {
    const rules = await api.get('/auto-label/rules', { params: { data_type: dtype } })
    labelRules.value = rules.items
  } else {
    labelRules.value = []
  }
  for (const n of form.nodes) if (n.type === 'auto_label') n.algorithm = ''
}

onMounted(async()=>{
  await loadDatasets()
  const input=routeInput()
  if(input){form.dataset_id=input.dataset_id;await onDatasetChange();form.input_version_id=input.version_id}
})
</script>

<style scoped>
.node-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
</style>
