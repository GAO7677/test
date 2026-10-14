<template>
  <div class="page">
    <div class="page-header">
      <h2><span>👥</span> 用户管理</h2>
      <p>账号 CRUD 与启用/禁用；新增与重置密码后用户下次登录须强制修改密码</p>
    </div>

    <div class="filter-bar">
      <el-input v-model="search" placeholder="搜索用户名 / 姓名…" clearable style="width: 220px" @change="load(1)" />
      <div style="margin-left: auto">
        <el-button v-if="auth.isSysAdmin" type="primary" @click="openEdit()">➕ 新增用户</el-button>
      </div>
    </div>

    <div class="table-card" v-loading="loading">
      <el-table :data="rows" @sort-change="onSort">
        <el-table-column prop="username" label="用户名" width="120" sortable="custom" />
        <el-table-column prop="display_name" label="姓名" width="110" />
        <el-table-column label="角色" width="110">
          <template #default="{ row }">
            <el-tag size="small" effect="dark"
                    :type="{ system_admin: 'danger', admin: 'warning', visitor: 'info' }[row.role]">{{ row.role_name }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="密级授权" width="200">
          <template #default="{ row }">
            <span v-if="row.role !== 'visitor'" style="color: var(--text-secondary)">不受限</span>
            <template v-else>
              <div style="font-size: 0.78rem">可见 ≤ {{ SECRET_LEVELS[row.view_level ?? 0] }}</div>
              <div style="font-size: 0.78rem">操作 {{ (row.operate_level ?? -1) < 0 ? '无' : '≤ ' + SECRET_LEVELS[row.operate_level] }}</div>
            </template>
          </template>
        </el-table-column>
        <el-table-column prop="department" label="部门" width="120" />
        <el-table-column prop="phone" label="电话" width="130" />
        <el-table-column label="状态" width="80">
          <template #default="{ row }">
            <el-tag size="small" effect="dark" :type="row.status === 'active' ? 'success' : 'info'">
              {{ row.status === 'active' ? '启用' : '禁用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="last_login_at" label="最后登录" width="160" sortable="custom">
          <template #default="{ row }">{{ row.last_login_at || '—' }}</template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="160" sortable="custom" />
        <el-table-column v-if="auth.isSysAdmin" label="操作" width="260" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="openEdit(row)">编辑</el-button>
            <el-button v-if="row.id !== auth.user?.id" link type="warning" size="small" @click="onResetPw(row)">重置密码</el-button>
            <el-button v-if="row.id !== auth.user?.id" link :type="row.status === 'active' ? 'info' : 'success'" size="small" @click="onToggleStatus(row)">
              {{ row.status === 'active' ? '禁用' : '启用' }}
            </el-button>
            <el-button v-if="row.id !== auth.user?.id" link type="danger" size="small" @click="onDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
      <div class="pager">
        <el-pagination background layout="total, sizes, prev, pager, next" :total="total"
                       v-model:current-page="page" v-model:page-size="pageSize"
                       :page-sizes="[10, 20, 50, 100]" @change="load()" />
      </div>
    </div>

    <el-dialog v-model="editVisible" :title="editRow ? '编辑用户' : '新增用户'" width="480px">
      <el-alert v-if="!editRow" type="info" :closable="false" style="margin-bottom: 12px"
                title="新用户将自动分配初始密码，首次登录须强制修改密码" />
      <el-form label-width="90px">
        <el-form-item label="用户名" required>
          <el-input v-model="form.username" :disabled="!!editRow" />
        </el-form-item>
        <el-form-item label="姓名"><el-input v-model="form.display_name" /></el-form-item>
        <el-form-item label="角色" required>
          <el-radio-group v-model="form.role">
            <el-radio-button value="system_admin">系统管理员</el-radio-button>
            <el-radio-button value="admin">管理员</el-radio-button>
            <el-radio-button value="visitor">访客</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <template v-if="form.role === 'visitor'">
          <el-form-item label="可见密级">
            <el-select v-model="form.view_level" style="width: 100%">
              <el-option v-for="(s, i) in SECRET_LEVELS" :key="i" :label="`≤ ${s}（含以下各级）`" :value="i" />
            </el-select>
            <div class="perm-tip">可浏览、预览、下载该密级及以下的数据</div>
          </el-form-item>
          <el-form-item label="可操作密级">
            <el-select v-model="form.operate_level" style="width: 100%">
              <el-option label="无（仅查看）" :value="-1" />
              <el-option v-for="(s, i) in SECRET_LEVELS" :key="i" :label="`≤ ${s}（含以下各级）`" :value="i"
                         :disabled="i > form.view_level" />
            </el-select>
            <div class="perm-tip">可对该密级及以下的数据执行清洗/标注/转换/采样/集成/评估/工作流（不能高于可见密级）</div>
          </el-form-item>
        </template>
        <el-alert v-else type="info" :closable="false" style="margin-bottom: 12px"
                  title="管理员角色不受密级限制，可访问与操作全部密级数据" />
        <el-form-item label="部门"><el-input v-model="form.department" /></el-form-item>
        <el-form-item label="电话"><el-input v-model="form.phone" /></el-form-item>
        <el-form-item label="邮箱"><el-input v-model="form.email" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="onSave">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api from '../api'
import { useAuthStore } from '../stores/auth'
import { SECRET_LEVELS } from '../utils'

const auth = useAuthStore()
const rows = ref([])
const total = ref(0)
const loading = ref(false)
const search = ref('')
const page = ref(1)
const pageSize = ref(20)
const sort = ref('')

async function load(p) {
  if (p) page.value = p
  loading.value = true
  try {
    const data = await api.get('/users', { params: { page: page.value, page_size: pageSize.value, search: search.value, sort: sort.value } })
    rows.value = data.items
    total.value = data.total
  } finally { loading.value = false }
}
function onSort({ prop, order }) {
  sort.value = order ? (order === 'descending' ? '-' : '') + prop : ''
  load()
}

const editVisible = ref(false)
const editRow = ref(null)
const saving = ref(false)
const form = reactive({ username: '', display_name: '', role: 'visitor', department: '', phone: '', email: '', view_level: 0, operate_level: -1 })

function openEdit(row) {
  editRow.value = row || null
  Object.assign(form, row
    ? { username: row.username, display_name: row.display_name, role: row.role, department: row.department, phone: row.phone, email: row.email, view_level: row.view_level ?? 0, operate_level: row.operate_level ?? -1 }
    : { username: '', display_name: '', role: 'visitor', department: '', phone: '', email: '', view_level: 0, operate_level: -1 })
  editVisible.value = true
}

async function onSave() {
  if (!editRow.value && !form.username) return ElMessage.error('用户名不能为空')
  saving.value = true
  try {
    if (editRow.value) await api.put(`/users/${editRow.value.id}`, form)
    else await api.post('/users', form)
    ElMessage.success('已保存')
    editVisible.value = false
    load()
  } catch { /* 已提示 */ } finally { saving.value = false }
}

async function onResetPw(row) {
  await ElMessageBox.confirm(`确认重置「${row.username}」的密码？重置后该用户下次登录须强制修改密码。`, '重置密码', { type: 'warning' })
  const r = await api.put(`/users/${row.id}/password`)
  ElMessage.success(r.message)
}

async function onToggleStatus(row) {
  const target = row.status === 'active' ? 'disabled' : 'active'
  await api.put(`/users/${row.id}/status`, { status: target })
  ElMessage.success(target === 'active' ? '已启用' : '已禁用')
  load()
}

async function onDelete(row) {
  await ElMessageBox.confirm(`确认删除用户「${row.username}」？（软删除）`, '删除确认', { type: 'warning' })
  await api.delete(`/users/${row.id}`)
  ElMessage.success('已删除')
  load()
}

onMounted(() => load())
</script>

<style scoped>
.perm-tip { font-size: 0.75rem; color: var(--text-secondary); line-height: 1.4; margin-top: 4px; }
</style>
