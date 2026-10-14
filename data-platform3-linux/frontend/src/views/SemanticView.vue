<template>
  <div class="page">
    <div class="page-header">
      <h2><span>🧠</span> 统一语义管理</h2>
      <p>航空领域统一语义标准库，规范术语定义、同义词映射与计量单位，支持术语标准化检查与一键纠正</p>
    </div>

    <div class="stat-cards">
      <div class="stat-card"><div class="stat-label">术语总数</div><div class="stat-value">{{ total }}</div></div>
      <div class="stat-card"><div class="stat-label">现行术语</div><div class="stat-value success">{{ activeCount }}</div></div>
    </div>

    <el-tabs v-model="tab">
      <el-tab-pane label="术语库" name="terms">
        <div class="filter-bar">
          <el-input v-model="search" placeholder="搜索术语 / 同义词 / 英文缩写…" clearable style="width: 260px" @change="load(1)" />
          <div style="margin-left: auto">
            <el-button v-if="auth.canWrite" type="primary" @click="openEdit()">➕ 新增术语</el-button>
          </div>
        </div>
        <div class="table-card" v-loading="loading">
          <el-table :data="rows">
            <el-table-column prop="term" label="术语" width="130" />
            <el-table-column prop="en_name" label="英文 / 缩写" width="190" show-overflow-tooltip />
            <el-table-column prop="definition" label="定义" min-width="240" show-overflow-tooltip />
            <el-table-column label="同义词" width="150">
              <template #default="{ row }">
                <el-tag v-for="s in row.synonyms" :key="s" size="small" effect="plain" style="margin: 0 4px 4px 0">{{ s }}</el-tag>
                <span v-if="!row.synonyms?.length" style="color: #64748b">—</span>
              </template>
            </el-table-column>
            <el-table-column prop="unit" label="单位" width="60" />
            <el-table-column prop="category" label="分类" width="100" />
            <el-table-column prop="secret_level" label="密级" width="70" />
            <el-table-column prop="status" label="状态" width="80">
              <template #default="{ row }">
                <el-tag size="small" effect="dark" :type="row.status === '现行' ? 'success' : row.status === '待审核' ? 'warning' : 'info'">{{ row.status }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column v-if="auth.canWrite" label="操作" width="110">
              <template #default="{ row }">
                <el-button link type="primary" size="small" @click="openEdit(row)">编辑</el-button>
                <el-button link type="danger" size="small" @click="onDelete(row)">删除</el-button>
              </template>
            </el-table-column>
            <template #empty><div class="empty-tip">无匹配术语</div></template>
          </el-table>
          <div class="pager">
            <el-pagination background layout="total, prev, pager, next" :total="total"
                           v-model:current-page="page" :page-size="20" @current-change="load()" />
          </div>
        </div>
      </el-tab-pane>

      <el-tab-pane label="术语标准化检查" name="check">
        <div class="table-card">
          <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0">
            扫描文本中的非标准用词（同义词命中项），支持一键替换为标准术语
          </p>
          <el-input v-model="checkText" type="textarea" :rows="5"
                    placeholder="粘贴待检查文本，如：攻角与偏流角随 M数 变化的关系曲线…" />
          <div style="margin: 12px 0">
            <el-button type="primary" @click="onCheck">🔍 开始检查</el-button>
            <el-button v-if="checkHits.length && auth.canWrite" type="warning" @click="onReplace">⚡ 一键替换为标准术语</el-button>
          </div>
          <el-table v-if="checkHits.length" :data="checkHits" size="small" style="margin-bottom: 12px">
            <el-table-column prop="synonym" label="命中同义词" width="140" />
            <el-table-column prop="standard" label="标准术语" width="140" />
            <el-table-column prop="en_name" label="英文" width="200" />
            <el-table-column prop="category" label="分类" width="120" />
            <el-table-column prop="count" label="出现次数" width="90" />
          </el-table>
          <el-alert v-else-if="checked" type="success" :closable="false" title="未发现非标准用词" />
          <div v-if="replacedText" style="margin-top: 12px">
            <div style="color: #34d399; font-size: 0.85rem; margin-bottom: 6px">替换结果：</div>
            <el-input v-model="replacedText" type="textarea" :rows="5" readonly />
          </div>
        </div>
      </el-tab-pane>
    </el-tabs>

    <!-- 新增/编辑术语 -->
    <el-dialog v-model="editVisible" :title="editRow ? '编辑术语' : '新增术语'" width="560px">
      <el-form label-width="100px">
        <el-form-item label="术语名称" required><el-input v-model="form.term" placeholder="如：迎角" /></el-form-item>
        <el-form-item label="英文/缩写"><el-input v-model="form.en_name" placeholder="如：Angle of Attack (AOA)" /></el-form-item>
        <el-form-item label="定义" required><el-input v-model="form.definition" /></el-form-item>
        <el-form-item label="同义词">
          <el-select v-model="form.synonyms" multiple allow-create filterable default-first-option
                     placeholder="输入后回车添加" style="width: 100%" />
        </el-form-item>
        <el-form-item label="单位"><el-input v-model="form.unit" style="width: 120px" /></el-form-item>
        <el-form-item label="分类">
          <el-select v-model="form.category" allow-create filterable style="width: 100%">
            <el-option v-for="c in ['飞行动力学','飞行性能','空气动力学','动力装置','雷达特性','雷达数据','数据质量','数据工程']"
                       :key="c" :label="c" :value="c" />
          </el-select>
        </el-form-item>
        <el-form-item label="密级">
          <el-radio-group v-model="form.secret_level">
            <el-radio-button v-for="s in ['绝密','机密','秘密','内部']" :key="s" :value="s">{{ s }}</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="状态">
          <el-radio-group v-model="form.status">
            <el-radio-button value="现行">现行</el-radio-button>
            <el-radio-button value="待审核">待审核</el-radio-button>
            <el-radio-button value="废止">废止</el-radio-button>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="onSave">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const tab = ref('terms')
const rows = ref([])
const total = ref(0)
const loading = ref(false)
const search = ref('')
const page = ref(1)

const activeCount = computed(() => rows.value.filter((r) => r.status === '现行').length)

async function load(p) {
  if (p) page.value = p
  loading.value = true
  try {
    const data = await api.get('/semantic/terms', { params: { page: page.value, page_size: 20, search: search.value } })
    rows.value = data.items
    total.value = data.total
  } finally { loading.value = false }
}

const editVisible = ref(false)
const editRow = ref(null)
const saving = ref(false)
const form = reactive({ term: '', en_name: '', definition: '', synonyms: [], unit: '', category: '', secret_level: '内部', status: '现行' })

function openEdit(row) {
  editRow.value = row || null
  Object.assign(form, row
    ? { term: row.term, en_name: row.en_name, definition: row.definition, synonyms: [...(row.synonyms || [])], unit: row.unit, category: row.category, secret_level: row.secret_level, status: row.status }
    : { term: '', en_name: '', definition: '', synonyms: [], unit: '', category: '', secret_level: '内部', status: '现行' })
  editVisible.value = true
}

async function onSave() {
  if (!form.term || !form.definition) return ElMessage.error('术语名称与定义不能为空')
  saving.value = true
  try {
    if (editRow.value) await api.put(`/semantic/terms/${editRow.value.id}`, form)
    else await api.post('/semantic/terms', form)
    ElMessage.success('已保存')
    editVisible.value = false
    load()
  } catch { /* 已提示 */ } finally { saving.value = false }
}

async function onDelete(row) {
  await ElMessageBox.confirm(`确认删除术语「${row.term}」？`, '删除确认', { type: 'warning' })
  await api.delete(`/semantic/terms/${row.id}`)
  ElMessage.success('已删除')
  load()
}

// ---- 标准化检查 ----
const checkText = ref('')
const checkHits = ref([])
const checked = ref(false)
const replacedText = ref('')

async function onCheck() {
  if (!checkText.value.trim()) return ElMessage.error('请输入待检查文本')
  const data = await api.post('/semantic/check', { text: checkText.value })
  checkHits.value = data.hits
  checked.value = true
  replacedText.value = ''
}

async function onReplace() {
  const data = await api.post('/semantic/replace', { text: checkText.value })
  replacedText.value = data.text
  ElMessage.success(`已替换 ${data.replaced} 处非标准用词`)
}

onMounted(() => load())
</script>
