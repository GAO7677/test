<template>
  <div class="page">
    <div class="page-header">
      <h2><span>⚙️</span> 系统配置</h2>
      <p>登录安全策略配置（仅系统管理员），存储于 system_settings 表，修改即时生效</p>
    </div>

    <div class="table-card" style="max-width: 680px" v-loading="loading">
      <el-form label-width="200px" label-position="left">
        <el-form-item label="连续登录失败锁定阈值">
          <el-input-number v-model="form.max_login_attempts" :min="1" :max="20" />
          <span class="hint">次失败后锁定账号</span>
        </el-form-item>
        <el-form-item label="账号锁定时长">
          <el-input-number v-model="form.lockout_duration_minutes" :min="1" :max="1440" />
          <span class="hint">分钟</span>
        </el-form-item>
        <el-form-item label="会话超时时间">
          <el-input-number v-model="form.session_timeout_minutes" :min="5" :max="1440" />
          <span class="hint">分钟无操作自动登出</span>
        </el-form-item>
        <el-form-item label="允许并发会话">
          <el-switch v-model="form.allow_concurrent_sessions" />
          <span class="hint">关闭后新登录将挤下线旧会话</span>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="saving" @click="onSave">保存配置</el-button>
          <el-button @click="load">还原</el-button>
        </el-form-item>
      </el-form>
    </div>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api from '../api'

const loading = ref(false)
const saving = ref(false)
const form = reactive({
  max_login_attempts: 5,
  lockout_duration_minutes: 30,
  session_timeout_minutes: 120,
  allow_concurrent_sessions: true,
})

async function load() {
  loading.value = true
  try {
    Object.assign(form, await api.get('/settings'))
  } finally { loading.value = false }
}

async function onSave() {
  saving.value = true
  try {
    await api.put('/settings', { ...form })
    ElMessage.success('配置已保存，即时生效')
  } catch { /* 已提示 */ } finally { saving.value = false }
}

onMounted(load)
</script>

<style scoped>
.hint { margin-left: 12px; color: #64748b; font-size: 0.8rem; }
</style>
